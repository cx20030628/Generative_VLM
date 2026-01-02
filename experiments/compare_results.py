import pandas as pd
import matplotlib.pyplot as plt


def generate_comparison_report(baseline_results, student_results, teacher_results):
    """
    Compare CNN Baseline, Distilled Student (CNN-Attention) and Teacher (Rationale-VLM)
    """
    data = {
        'Model': ['CNN Baseline', 'CNN-Attention (Distilled)', 'Rationale-VLM (Teacher)'],
        'Accuracy': [baseline_results['acc'], student_results['acc'], teacher_results['acc']],
        'Robustness (RQR)': [baseline_results['rqr'], student_results['rqr'], teacher_results['rqr']],
        'Inference Time (ms)': [15, 22, 450]  # Example inference times
    }

    df = pd.DataFrame(data)
    print("--- Overall Performance Comparison ---")
    print(df)

    # draw comparison chart
    df.set_index('Model')[['Accuracy', 'Robustness (RQR)']].plot(kind='bar')
    plt.title("Performance Gap: Baseline vs Student vs Teacher")
    plt.savefig('data/processed/comparison_chart.png')