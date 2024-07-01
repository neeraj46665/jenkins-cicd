import mlflow
from sklearn import metrics
import os
import matplotlib.pyplot as plt

def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    # Close plot
    plt.close()
    return accuracy, f1, auc

def mlflow_logging(model, X, y, model_name):
    # Check if there is an active run and end it if necessary
    if mlflow.active_run():
        mlflow.end_run()
    
    # Start a new MLflow run
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("model_name", model_name)
        
        pred = model.predict(X)
        accuracy, f1, auc = eval_metrics(y, pred)
        
        # Log the metrics
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("F1-score", f1)
        mlflow.log_metric("AUC", auc)
        
        # Logging artifacts and model
        mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.sklearn.log_model(model, artifact_path="models")
