from sklearn.calibration import calibration_curve
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from collections import Counter
import numpy as np
import time

from .alignment import Levenshtein

class MutliClassEval:
    def __init__(self, preds, labels):
        self.preds, self.labels = preds, labels
        self.num_classes = len(preds[0])

    def misaligned_eval(self):        
        results = np.zeros(5)
        for conv_pred, lab in zip(self.preds, self.labels):
            decision = np.argmax(conv_pred, axis=-1)            
            errors, decisions = Levenshtein.lev_dist(decision, lab)
            err_cou = Counter(decisions)
            subs, inserts, deletes = (err_cou[i] for i in ['r', 'i', 'd'])
            
            results += [len(lab), errors, subs, inserts, deletes]

        print(f"WER:{results[1]/results[0]:.3f}  replace:{results[2]/results[0]:.3f}  ",
              f"inserts: {results[3]/results[0]:.3f}  deletion: {results[4]/results[0]:.3f}")
            
    def classification_report(self, detail=False, names=None):
        decision = np.argmax(self.preds, axis=-1)
        labels = self.labels
        
        if (names and detail):
            decision = [names[i] for i in decision]
            labels = [names[i] for i in labels]

        report = metrics.classification_report(labels, decision, digits=3)  
        
        rows = report.split('\n')
        header = rows[0]
        summary = '\n'.join(rows[-4:-1])
        
        print(header)
        if detail:
            #This is all done to be able to sort by F1 (can change index to sort by P,R or support)
            report = metrics.classification_report(labels, decision, digits=3, output_dict=True)  
            results = [[k, v['precision'], v['recall'], v['f1-score'], v['support']] \
                       for k, v in report.items() if k not in ['weighted avg', 'macro avg', 'accuracy']] 
            results.sort(key=lambda x: x[3], reverse=True)
            
            width=max([len(x[0]) for x in results])
            for row in results:
                print(f' {row[0]:>{width}} {row[1]:>9.3f} {row[2]:>9.3f} {row[3]:>9.3f} {row[4]:>9}')
            print()
        print(summary)
        
    def one_vs_all_eval(self):
        class_perfs = self._one_vs_all_eval()
        thresholds = np.array([act.op_point[0] for act in class_perfs])

        hits_dist, score_thresh = [], 0
        for pred, lab in zip(self.preds, self.labels):
            hits = np.array(pred) > thresholds
            hits_dist.append(sum(hits))
            if sum(hits) > 0:
                score_thresh += hits[lab]/sum(hits)
        
        print(f'accuracy using 1 vs all: {score_thresh/len(self.preds):.3f}')
        print(Counter(hits_dist))
        sns.histplot(hits_dist, binwidth=1, binrange=[-0.5, 4])
        plt.show()

    def plot_class_curves(self):
        class_perfs = self._one_vs_all_eval()
        thresholds = np.array([act.op_point[0] for act in class_perfs])

        sns.set_theme()
        precisions = np.array([act.op_point[2] for act in class_perfs])
        recalls = np.array([act.op_point[3] for act in class_perfs])
        frequencies = np.array([sum(act.labels) for act in class_perfs])
        colors = cm.rainbow(np.linspace(0, 1, len(class_perfs)))

        ax = iso_F1()
        ax.scatter(precisions, recalls, color=colors, marker='x')
        plt.show()
    
        fig, ax = plt.subplots(figsize=(10,0.5))
        ax.scatter(frequencies, np.zeros(len(self.class_perfs)), color=colors, marker='x')
        ax.get_yaxis().set_ticks([])
        ax.set_xlabel('frequency of training points')
        ax.set_xscale("log")
        ax.set_xlim([0.5,2*max(frequencies)])
        plt.show()

    def _one_vs_all_eval(self):
        classes = []
        for k in range(self.num_classes):
            class_preds, class_labels = single_class(self.preds, self.labels, k)
            classes.append(BinaryEval(class_preds, class_labels))
        return classes

    def __getitem__(self, k):
        return self.class_perfs[k]
        
def single_class(preds, labels, k):
    class_preds, class_labels = [], []
    for pred, lab in zip(preds, labels):
        class_preds.append(pred[k])
        class_labels.append(int(lab == k))
    return class_preds, class_labels

class BinaryEval:
    def __init__(self, preds, labels):
        self.preds = preds
        self.labels = labels

    def plot_PR(self):
        precision, recall, thresholds = self.PR()
        ax = iso_F1()
        ax.plot(recall, precision)

        threshold_markings = []
        for p,r,t in zip(precision, recall, thresholds):
            if all((p-p_2)**2 + (r-r_2)**2 > 0.04  for (p_2, r_2) in threshold_markings):
                threshold_markings.append((p,r))
                ax.text(r+0.02, p+0.005, f'{t:.3f}', color='r')
                ax.plot(r, p, 'rx')

    def plot_calib(self):
        prob_pred, prob_true = self.calibration()
        fig, ax = plt.subplots()
        x = np.linspace(0,1,50)
        ax.plot(x, x)
        ax.plot(prob_pred, prob_true, 'x')
        ax.set_xlim(0,1), ax.set_ylim(0,1.02)

    def PR(self):
        precision, recall, thresholds = metrics.precision_recall_curve(self.labels, self.preds)
        return (precision, recall, thresholds)

    def calibration(self):
        prob_true, prob_pred = calibration_curve(self.labels, self.preds, n_bins=10, strategy='quantile')
        return (prob_pred, prob_true)
    
    @property
    def ECE(self):
        prob_pred, prob_true = self.calibration()
        """
        x = np.linspace(0,1,50)
        plt.plot(x, x)
        plt.plot(prob_pred, prob_true, 'x')
        plt.show()
        plt.xlim(0,1), plt.ylim(0,1.02)
        """
        return sum([abs(i-j) for i, j in zip(prob_true, prob_pred)])

    @property
    def op_point(self):
        precision, recall, thresholds = self.PR()
        F1 = [calc_F1(P, R) for P, R in zip(precision, recall)]
        operating_point = max(zip(thresholds, F1, precision, recall), key=lambda x: x[1])
        return operating_point

    def __repr__(self):
        T, F1, P, R = self.op_point
        output = f'T:{T:.3f}   F1:{F1:.3f}   P:{P:.3f}   R:{R:.3f}'
        return output
        
def iso_F1():
    fig, ax = plt.subplots()
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = ax.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        ax.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    ax.set_xlim(0,1), ax.set_ylim(0,1.02)
    ax.set_xlabel('Recall'), ax.set_ylabel('Precision') 
    return ax

def calc_F1(P, R):
    return (2*P*R)/(P+R)
