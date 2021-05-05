#!/usr/bin/env python

# This file contains functions for evaluating algorithms for the 2021 PhysioNet/
# Computing in Cardiology Challenge. You can run it as follows:
#
#   python evaluate_model.py labels outputs scores.csv
#
# where 'labels' is a directory containing files with the labels, 'outputs' is a
# directory containing files with the outputs from your model, and 'scores.csv'
# (optional) is a collection of scores for the algorithm outputs.
#
# Each file of labels or outputs must have the format described on the Challenge
# webpage. The scores for the algorithm outputs include the area under the
# receiver-operating characteristic curve (AUROC), the area under the recall-
# precision curve (AUPRC), accuracy (fraction of correct recordings), macro F-
# measure, and the Challenge metric, which assigns different weights to
# different misclassification errors.

import os, os.path, sys, numpy as np
from helper_code import get_labels, is_finite_number, load_header, load_outputs

def evaluate_model(label_directory, output_directory):
    # Identify the weights and the SNOMED CT code for the sinus rhythm class.
    weights_file = 'weights.csv'
    sinus_rhythm = set(['426783006'])

    # Load the scored classes and the weights for the Challenge metric.
    print('Loading weights...')
    classes, weights = load_weights(weights_file)

    # Load the label and output files.
    print('Loading label and output files...')
    label_files, output_files = find_challenge_files(label_directory, output_directory)
    labels = load_labels(label_files, classes)
    binary_outputs, scalar_outputs = load_classifier_outputs(output_files, classes)

    # Evaluate the model by comparing the labels and outputs.
    print('Evaluating model...')

    print('- AUROC and AUPRC...')
    auroc, auprc, auroc_classes, auprc_classes = compute_auc(labels, scalar_outputs)

    print('- Accuracy...')
    accuracy = compute_accuracy(labels, binary_outputs)

    print('- F-measure...')
    f_measure, f_measure_classes = compute_f_measure(labels, binary_outputs)

    print('- Challenge metric...')
    challenge_metric = compute_challenge_metric(weights, labels, binary_outputs, classes, sinus_rhythm)

    print('Done.')

    # Return the results.
    return classes, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_measure_classes, challenge_metric

# Find Challenge files.
def find_challenge_files(label_directory, output_directory):
    label_files = list()
    output_files = list()
    for label_file in sorted(os.listdir(label_directory)):
        label_file_path = os.path.join(label_directory, label_file) # Full path for label file
        if os.path.isfile(label_file_path) and label_file.lower().endswith('.hea') and not label_file.lower().startswith('.'):
            root, ext = os.path.splitext(label_file)
            output_file = root + '.csv'
            output_file_path = os.path.join(output_directory, output_file) # Full path for corresponding output file
            if os.path.isfile(output_file_path):
                label_files.append(label_file_path)
                output_files.append(output_file_path)
            else:
                raise IOError('Output file {} not found for label file {}.'.format(output_file, label_file))

    if label_files and output_files:
        return label_files, output_files
    else:
        raise IOError('No label or output files found.')

# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))
    row_lengths = set(len(table[i])-1 for i in range(num_rows))
    if len(row_lengths)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(row_lengths)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_finite_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

# Load weights.
def load_weights(weight_file):
    # Load the table with the weight matrix.
    rows, cols, values = load_table(weight_file)

    # Split the equivalent classes.
    rows = [set(row.split('|')) for row in rows]
    cols = [set(col.split('|')) for col in cols]
    assert(rows == cols)

    # Identify the classes and the weight matrix.
    classes = rows
    weights = values

    return classes, weights

# Load labels from header/label files.
def load_labels(label_files, classes):
    # The labels should have the following form:
    #
    # Dx: label_1, label_2, label_3
    #
    num_recordings = len(label_files)
    num_classes = len(classes)

    # Use one-hot encoding for the labels.
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)

    # Iterate over the recordings.
    for i in range(num_recordings):
        header = load_header(label_files[i])
        y = set(get_labels(header))
        for j, x in enumerate(classes):
            if x & y:
                labels[i, j] = 1

    return labels

# Load outputs from output files.
def load_classifier_outputs(output_files, classes):
    # The outputs should have the following form:
    #
    # #Record ID
    # diagnosis_1, diagnosis_2, diagnosis_3
    #           0,           1,           1
    #        0.12,        0.34,        0.56
    #
    num_recordings = len(output_files)
    num_classes = len(classes)

    # Use one-hot encoding for the outputs.
    binary_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    scalar_outputs = np.zeros((num_recordings, num_classes), dtype=np.float64)

    # Iterate over the recordings.
    for i in range(num_recordings):
        recording_id, recording_classes, recording_binary_outputs, recording_scalar_outputs = load_outputs(output_files[i])

        # Allow for equivalent classes and sanitize classifier outputs.
        recording_classes = [set(entry.split('|')) for entry in recording_classes]
        recording_binary_outputs = [1 if entry in ('1', 'True', 'true', 'T', 't') else 0 for entry in recording_binary_outputs]
        recording_scalar_outputs = [float(entry) if is_finite_number(entry) else 0 for entry in recording_scalar_outputs]

        # Allow for unordered/reordered and equivalent classes.
        for j, x in enumerate(classes):
            binary_values = list()
            scalar_values = list()
            for k, y in enumerate(recording_classes):
                if x & y:
                    binary_values.append(recording_binary_outputs[k])
                    scalar_values.append(recording_scalar_outputs[k])
            if binary_values:
                binary_outputs[i, j] = any(binary_values) # Define a class as positive if any of the equivalent classes is positive.
            if scalar_values:
                scalar_outputs[i, j] = np.mean(scalar_values) # Define the scalar value of a class as the mean value of the scalar values across equivalent classes.

    return binary_outputs, scalar_outputs

# Compute recording-wise accuracy.
def compute_accuracy(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :]==outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)

# Compute confusion matrices.
def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1.0/normalization
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')

    return A

# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float('nan')

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, f_measure

# Compute macro AUROC and macro AUPRC.
def compute_auc(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1]+1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k]==1)
        tn[0] = np.sum(labels[:, k]==0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j-1]
            fp[j] = fp[j-1]
            fn[j] = fn[j-1]
            tn[j] = tn[j-1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float('nan')
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float('nan')
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float('nan')

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds-1):
            auroc[k] += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
            auprc[k] += (tpr[j+1] - tpr[j]) * ppv[j+1]

    # Compute macro AUROC and macro AUPRC across classes.
    if np.any(np.isfinite(auroc)):
        macro_auroc = np.nanmean(auroc)
    else:
        macro_auroc = float('nan')
    if np.any(np.isfinite(auprc)):
        macro_auprc = np.nanmean(auprc)
    else:
        macro_auprc = float('nan')

    return macro_auroc, macro_auprc, auroc, auprc

# Compute a modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/normalization

    return A

# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(weights, labels, outputs, classes, sinus_rhythm):
    num_recordings, num_classes = np.shape(labels)
    if sinus_rhythm in classes:
        sinus_rhythm_index = classes.index(sinus_rhythm)
    else:
        raise ValueError('The sinus rhythm class is not available.')

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the sinus rhythm class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, sinus_rhythm_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score

if __name__ == '__main__':
    classes, auroc, auprc, auroc_classes, auprc_classes, accuracy, f_measure, f_measure_classes, challenge_metric = evaluate_model(sys.argv[1], sys.argv[2])
    output_string = 'AUROC,AUPRC,Accuracy,F-measure,Challenge metric\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(auroc, auprc, accuracy, f_measure, challenge_metric)
    class_output_string = 'Classes,{}\nAUROC,{}\nAUPRC,{}\nF-measure,{}'.format(
        ','.join('|'.join(sorted(x)) for x in classes),
        ','.join('{:.3f}'.format(x) for x in auroc_classes),
        ','.join('{:.3f}'.format(x) for x in auprc_classes),
        ','.join('{:.3f}'.format(x) for x in f_measure_classes))

    if len(sys.argv) == 3:
        print(output_string)
    elif len(sys.argv) == 4:
        with open(sys.argv[3], 'w') as f:
            f.write(output_string)
    elif len(sys.argv) == 5:
        with open(sys.argv[3], 'w') as f:
            f.write(output_string)
        with open(sys.argv[4], 'w') as f:
            f.write(class_output_string)
