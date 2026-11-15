"""Summarize completed Qwen Text-to-SQL benchmark without changing raw data.

Run from project root (LM Studio can be OFF):
    python eval/analyze_qwen_results.py

Requires: pandas, matplotlib.
Generates CSV tables and PNG figures under eval/results/analysis/.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / 'eval' / 'results' / 'qwen_7b_results.csv'
DEFAULT_AUDIT = ROOT / 'eval' / 'results' / 'timeout_recheck.csv'
DEFAULT_OUTPUT = ROOT / 'eval' / 'results' / 'analysis'
CONDITIONS = ('schema_only', 'with_semantic_layer')


def error_subtype(row):
    status = row['status']
    error = row['error'].lower()
    if status == 'execution_error':
        if 'exceeded 12 seconds' in error or 'timed out' in error:
            return 'timeout'
        if 'binderexception' in error or 'binder error' in error:
            return 'sql_binder_error'
        if 'parserexception' in error or 'parser error' in error:
            return 'sql_parser_error'
        return 'other_execution_error'
    return status


def bar_value_labels(ax, bars, decimals=0):
    for rect in bars:
        h = rect.get_height()
        if h:
            ax.annotate(
                f'{h:.{decimals}f}',
                (rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=9,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.results.is_file():
        parser.error(f'Results CSV not found: {args.results}')
    results = pd.read_csv(args.results, keep_default_na=False, encoding='utf-8-sig')
    required = {'id', 'condition', 'status', 'error'}
    missing = required.difference(results.columns)
    if missing:
        parser.error(f'Missing results columns: {sorted(missing)}')
    if results.duplicated(['id', 'condition']).any():
        parser.error('Duplicate (id, condition) records found; resolve before analysis.')
    unexpected = set(results['condition']) - set(CONDITIONS)
    if unexpected:
        parser.error(f'Unexpected condition(s): {sorted(unexpected)}')

    expected = pd.MultiIndex.from_product([
        sorted(results['id'].unique()), CONDITIONS,
    ], names=['id', 'condition'])
    if len(results) != 100 or results['id'].nunique() != 50 or len(expected.difference(
        pd.MultiIndex.from_frame(results[['id', 'condition']])
    )) != 0:
        print('WARNING: Not a complete 50-question x 2-condition dataset. Outputs are preliminary.')

    results = results.copy()
    results['error_subtype'] = results.apply(error_subtype, axis=1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The original benchmark result is the PRIMARY outcome. No records are edited.
    summaries = []
    for condition in CONDITIONS:
        sub = results.loc[results['condition'] == condition]
        total = len(sub)
        original_correct = int((sub['status'] == 'correct').sum())
        counts = sub['error_subtype'].value_counts()
        summaries.append({
            'condition': condition,
            'total': total,
            'recorded_correct': original_correct,
            'recorded_end_to_end_pct': round(100 * original_correct / total, 1) if total else 0,
            'timeouts': int(counts.get('timeout', 0)),
            'sql_binder_errors': int(counts.get('sql_binder_error', 0)),
            'invalid_sql': int(counts.get('invalid_sql', 0)),
            'blocked_sql': int(counts.get('blocked_sql', 0)),
            'wrong_result': int(counts.get('wrong_result', 0)),
            'llm_error': int(counts.get('llm_error', 0)),
            'other_errors': int(sum(v for k, v in counts.items() if k not in {
                'correct', 'timeout', 'sql_binder_error', 'invalid_sql',
                'blocked_sql', 'wrong_result', 'llm_error',
            })),
        })
    summary = pd.DataFrame(summaries)

    # Independent timeout audit: supporting evidence, never overwrites raw results.
    audit_present = args.audit.is_file()
    if audit_present:
        audit = pd.read_csv(args.audit, keep_default_na=False, encoding='utf-8-sig')
        required_audit = {'id', 'condition', 'recheck_outcome'}
        if not required_audit.issubset(audit.columns):
            parser.error(f'Audit missing fields: {sorted(required_audit.difference(audit.columns))}')
        if audit.duplicated(['id', 'condition']).any():
            parser.error('Duplicate (id, condition) rows in timeout audit.')
        audit['recheck_outcome'] = audit['recheck_outcome'].astype(str)
        audit_summary = (
            audit.groupby(['condition', 'recheck_outcome'])
            .size().rename('count').reset_index()
        )
        audit_summary.to_csv(args.output_dir / 'timeout_audit_summary.csv', index=False)
        verified = audit.loc[audit['recheck_outcome'] == 'matches_reference', ['id','condition']]
        # Only count records originally timed out; don't double-count pre-existing correct rows.
        verified = verified.merge(
            results[['id', 'condition', 'error_subtype']], on=['id', 'condition'], how='left', validate='one_to_one',
        )
        verified = verified.loc[verified['error_subtype'] == 'timeout']
        for i, row in summary.iterrows():
            added = int((verified['condition'] == row['condition']).sum())
            summary.loc[i, 'additional_timeout_matches'] = added
            summary.loc[i, 'recheck_supported_correct'] = int(row['recorded_correct']) + added
            summary.loc[i, 'recheck_supported_pct'] = round(
                100 * (int(row['recorded_correct']) + added) / int(row['total']), 1
            ) if row['total'] else 0
    else:
        audit = None
        print(f'NOTE: Timeout audit not found at {args.audit}. Skipping supplementary columns.')

    summary.to_csv(args.output_dir / 'condition_summary.csv', index=False)
    errors = results.loc[results['status'] != 'correct'].copy()
    if audit_present:
        errors = errors.merge(audit[['id', 'condition', 'recheck_outcome']],
                              how='left', on=['id','condition'], validate='one_to_one')
    errors.to_csv(args.output_dir / 'error_details.csv', index=False)

    # Make a per-question paired status table.
    by_question = results.pivot(index='id', columns='condition', values='status').reset_index()
    by_question.columns.name = None
    by_question.to_csv(args.output_dir / 'paired_question_status.csv', index=False)

    # Plot 1: the original end-to-end accuracy. Keep it separate from recheck results.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    display = ['Schema only', 'With semantic layer']
    vals = summary['recorded_end_to_end_pct'].to_list()
    bars = ax.bar(display, vals, width=0.55)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Recorded end-to-end correctness (%)')
    ax.set_title('Qwen 2.5 7B — Original 50-question evaluation')
    bar_value_labels(ax, bars, decimals=1)
    ax.text(0.5, -0.17, 'Timeouts and rejected queries count as failures',
            ha='center', transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output_dir / 'recorded_accuracy.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    # Plot 2: disaggregated failure types (doesn't include correct rows).
    types = ['timeout', 'sql_binder_error', 'wrong_result', 'blocked_sql',
             'invalid_sql', 'llm_error', 'sql_parser_error', 'other_execution_error']
    breakdown = (
        errors.groupby(['condition','error_subtype']).size()
        .unstack(fill_value=0).reindex(index=CONDITIONS, columns=types, fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 4.9))
    bottoms = [0, 0]
    for kind in types:
        counts = breakdown[kind].to_list()
        if not any(counts):
            continue
        ax.bar(display, counts, bottom=bottoms, label=kind.replace('_',' '))
        bottoms = [a+b for a,b in zip(bottoms, counts)]
    ax.set_ylabel('Number of unsuccessful evaluations')
    ax.set_title('Failure categories — original benchmark')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, max(bottoms) + 3)
    fig.tight_layout()
    fig.savefig(args.output_dir / 'error_distribution.png', dpi=170, bbox_inches='tight')
    plt.close(fig)

    print('\nORIGINAL BENCHMARK SUMMARY')
    print(summary[['condition', 'total', 'recorded_correct', 'recorded_end_to_end_pct']].to_string(index=False))
    if audit_present:
        print('\nSUPPLEMENTARY TIMEOUT RECHECK (NOT A NEW CONTROLLED BENCHMARK)')
        print(summary[['condition', 'additional_timeout_matches', 'recheck_supported_correct',
                       'recheck_supported_pct']].to_string(index=False))
        print('\nTIMEOUT AUDIT')
        print(audit_summary.to_string(index=False))
    print('\nSaved reports to:', args.output_dir)
    print('Original evaluation and timeout audit CSV files were not modified.')
    print('Reference answers still require manual semantic review.')


if __name__ == '__main__':
    main()
