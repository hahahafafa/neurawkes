# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Export interval-level horizon predictions from a trained legacy Neurawkes model.
"""

import argparse
import csv
import json
import os

import numpy

import modules.controllers as controllers


def load_event_type_mapping(path):
    idx_to_label = {}
    with open(path, 'rb') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_to_label[int(row['type_event'])] = str(row['event_label'])
    return idx_to_label


def load_interval_rows(path):
    rows = []
    with open(path, 'rb') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['transition_index'] = int(row['transition_index'])
            row['prefix_event_count'] = int(row['prefix_event_count'])
            row['event_indicator'] = int(row['event_indicator'])
            row['observed_duration'] = float(row['observed_duration'])
            row['entry_time'] = float(row['entry_time'])
            row['exit_time'] = float(row['exit_time'])
            row['current_state'] = str(row['current_state'])
            row['next_state'] = str(row['next_state']) if row['next_state'] is not None else ''
            row['event_label'] = str(row['event_label'])
            rows.append(row)
    rows.sort(key=lambda item: (item['subject_id'], item['transition_index']))
    return rows


def load_event_sequences(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'rb') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['event_index'] = int(row['event_index'])
            row['type_event'] = int(row['type_event'])
            row['time_since_start'] = float(row['time_since_start'])
            row['time_since_last_event'] = float(row['time_since_last_event'])
            rows.append(row)
    rows.sort(key=lambda item: (item['subject_id'], item['event_index']))
    return rows


def group_by_subject(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['subject_id'], []).append(row)
    return grouped


def sanitize_label(label):
    out = []
    for ch in str(label):
        if ch.isalnum():
            out.append(ch)
        else:
            out.append('_')
    while '__' in ''.join(out):
        out = list(''.join(out).replace('__', '_'))
    return ''.join(out).strip('_')


def build_transition_lookup(idx_to_label):
    transition_lookup = {}
    next_states = set()
    for idx_type, label in idx_to_label.iteritems():
        parts = label.split('->')
        if len(parts) != 2:
            continue
        from_state = parts[0]
        to_state = parts[1]
        transition_lookup[idx_type] = (from_state, to_state, label)
        next_states.add(to_state)
    return transition_lookup, sorted(next_states)


def make_control(path_pre_train, dim_process):
    settings = {
        'model': 'conttime',
        'loss_type': 'prediction',
        'dim_process': numpy.int32(dim_process),
        'coef_l2': numpy.float32(0.0),
        'size_batch': numpy.int32(1),
        'optimizer': 'adam',
        'path_pre_train': path_pre_train,
        'learn_rate': numpy.float32(1e-3),
        'predict_lambda': False
    }
    return controllers.ControlNeuralHawkesAdaptiveBaseCTSM_time(settings)


def build_sequence_arrays(event_rows, dim_process):
    seq_len = len(event_rows)
    seq_type_event = numpy.zeros((seq_len + 1, 1), dtype=numpy.int32)
    seq_time_values = numpy.zeros((seq_len + 1, 1), dtype=numpy.float32)
    seq_type_event[0, 0] = numpy.int32(dim_process)
    for idx, row in enumerate(event_rows):
        seq_type_event[idx + 1, 0] = numpy.int32(row['type_event'])
        seq_time_values[idx + 1, 0] = numpy.float32(row['time_since_last_event'])
    return seq_type_event, seq_time_values


def export_predictions(args):
    idx_to_label = load_event_type_mapping(args.event_type_mapping)
    transition_lookup, next_states = build_transition_lookup(idx_to_label)
    interval_rows = load_interval_rows(args.interval_rows)
    event_rows = load_event_sequences(args.event_sequences)
    interval_by_subject = group_by_subject(interval_rows)
    events_by_subject = group_by_subject(event_rows)
    subject_ids = sorted(interval_by_subject.keys())

    control = make_control(args.file_pretrain, len(idx_to_label))
    if args.num_time_samples <= 0:
        raise ValueError('num_time_samples must be positive')
    time_diffs = numpy.linspace(
        float(args.horizon) / float(args.num_time_samples),
        float(args.horizon),
        int(args.num_time_samples)
    ).astype(numpy.float32)

    transition_keys = sorted(idx_to_label.keys())
    transition_fieldnames = [
        'p_transition_' + sanitize_label(idx_to_label[idx_type]) for idx_type in transition_keys
    ]
    next_state_fieldnames = [
        'p_next_' + sanitize_label(state) for state in next_states
    ]
    fieldnames = [
        'split',
        'subject_id',
        'transition_index',
        'prefix_event_count',
        'current_state',
        'observed_duration',
        'event_indicator',
        'true_event',
        'true_next_state',
        'included_in_horizon_metrics',
        'p_event',
        'pred_event',
        'pred_next_state',
        'pred_transition_label',
    ] + transition_fieldnames + next_state_fieldnames

    output_rows = []
    for subject_id in subject_ids:
        seq_type_event, seq_time_values = build_sequence_arrays(
            events_by_subject.get(subject_id, []),
            len(idx_to_label)
        )
        prob_over_prefix, time_prediction, type_prediction = control.model_predict(
            seq_type_event,
            seq_time_values,
            time_diffs
        )
        prob_over_prefix = numpy.asarray(prob_over_prefix)
        type_prediction = numpy.asarray(type_prediction)
        interval_items = interval_by_subject[subject_id]
        for interval_row in interval_items:
            prefix = int(interval_row['prefix_event_count'])
            if prefix < 0 or prefix >= prob_over_prefix.shape[0]:
                raise ValueError(
                    'Prefix index %s for subject %s is outside prediction range %s'
                    % (prefix, subject_id, prob_over_prefix.shape[0])
                )
            probs = numpy.asarray(prob_over_prefix[prefix, 0, :], dtype=numpy.float64)
            current_state = str(interval_row['current_state'])
            next_state_probs = {}
            p_event = 0.0
            pred_transition_label = 'NO_EVENT'
            pred_transition_prob = -1.0
            for idx_pos, idx_type in enumerate(transition_keys):
                label = idx_to_label[idx_type]
                prob = float(probs[idx_type])
                if idx_type in transition_lookup:
                    from_state, to_state, _ = transition_lookup[idx_type]
                    if from_state == current_state:
                        p_event += prob
                        next_state_probs[to_state] = next_state_probs.get(to_state, 0.0) + prob
                        if prob > pred_transition_prob:
                            pred_transition_prob = prob
                            pred_transition_label = label
            pred_event = int(p_event >= float(args.event_threshold))
            if next_state_probs:
                pred_next_state = max(next_state_probs, key=next_state_probs.get)
            else:
                pred_next_state = 'NO_EVENT'
            observed_duration = float(interval_row['observed_duration'])
            true_event = int(interval_row['event_indicator'] == 1 and observed_duration <= float(args.horizon))
            true_next_state = interval_row['next_state'] if true_event and interval_row['next_state'] else 'NO_EVENT'
            included = int(true_event == 1 or observed_duration >= float(args.horizon))

            output_row = {
                'split': args.tag_split,
                'subject_id': subject_id,
                'transition_index': int(interval_row['transition_index']),
                'prefix_event_count': prefix,
                'current_state': current_state,
                'observed_duration': observed_duration,
                'event_indicator': int(interval_row['event_indicator']),
                'true_event': true_event,
                'true_next_state': true_next_state,
                'included_in_horizon_metrics': included,
                'p_event': float(p_event),
                'pred_event': pred_event,
                'pred_next_state': pred_next_state,
                'pred_transition_label': pred_transition_label,
            }
            for idx_type in transition_keys:
                output_row['p_transition_' + sanitize_label(idx_to_label[idx_type])] = float(probs[idx_type])
            for state in next_states:
                output_row['p_next_' + sanitize_label(state)] = float(next_state_probs.get(state, 0.0))
            output_rows.append(output_row)

    with open(args.output_csv, 'wb') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    summary = {
        'tag_split': args.tag_split,
        'horizon': float(args.horizon),
        'event_threshold': float(args.event_threshold),
        'num_time_samples': int(args.num_time_samples),
        'num_rows': len(output_rows),
        'num_subjects': len(subject_ids),
        'output_csv': os.path.abspath(args.output_csv),
    }
    with open(args.output_csv + '.json', 'wb') as f:
        f.write(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Export interval-level Neurawkes predictions.')
    parser.add_argument('-fd', '--FileData', dest='file_data', required=True)
    parser.add_argument('-fp', '--FilePretrain', dest='file_pretrain', required=True)
    parser.add_argument('-ts', '--TagSplit', dest='tag_split', required=True, choices=['dev', 'test'])
    parser.add_argument('-ir', '--IntervalRows', dest='interval_rows', required=True)
    parser.add_argument('-es', '--EventSequences', dest='event_sequences', required=True)
    parser.add_argument('-em', '--EventTypeMapping', dest='event_type_mapping', required=True)
    parser.add_argument('-o', '--OutputCsv', dest='output_csv', required=True)
    parser.add_argument('-hz', '--Horizon', dest='horizon', type=float, default=1.0)
    parser.add_argument('-et', '--EventThreshold', dest='event_threshold', type=float, default=0.5)
    parser.add_argument('-nt', '--NumTimeSamples', dest='num_time_samples', type=int, default=200)
    args = parser.parse_args()
    export_predictions(args)


if __name__ == '__main__':
    main()
