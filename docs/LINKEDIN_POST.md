# LinkedIn post draft

Attach `docs/screenshot.png` (or a 20-second screen recording of clicking the two example buttons;
video gets noticeably more reach). Replace the two links before posting.

---

I built and deployed an end-to-end machine learning app that predicts which customers are about to cancel, and explains why. 📉➡️📈

Most ML portfolio projects stop at a notebook. I wanted to take one all the way to production.

🔍 The problem
Keeping a customer is far cheaper than winning a new one. If a retention team knows who is at risk, and what's driving it, they can act before the customer leaves.

🧠 The model
• Trained on 7,043 telecom customers (IBM Telco dataset)
• Compared Logistic Regression, Random Forest and Gradient Boosting with 5-fold cross-validation
• Gradient Boosting won: 0.84 ROC-AUC on a held-out test set
• Tuned the decision threshold on out-of-fold predictions instead of using the default 0.5. That lifted recall from 50% to 72%, so far fewer churners slip through
• Every prediction comes with its top risk drivers (e.g. "tenure = 2 months, +22 pts")

⚙️ The engineering
• FastAPI service with strict input validation and batch scoring
• Shared feature code for training and serving, so they can't drift apart
• Multi-stage Docker build that trains the model inside the image
• GitHub Actions CI/CD: lint → train → quality gate (fails if ROC-AUC < 0.80) → tests → Docker smoke test → auto-deploy
• Hosted on Render

💡 Biggest lesson: the threshold matters as much as the algorithm. All three models scored within 0.003 ROC-AUC of each other; choosing the right cut-off for the business cost changed recall by 22 points.

🔗 Live demo: https://YOUR-APP.onrender.com
💻 Code: https://github.com/YOUR_USERNAME/churn-predictor

Feedback welcome, especially on what you'd add next. I'm thinking drift monitoring and SHAP explanations.

#MachineLearning #DataScience #MLOps #Python #FastAPI #Docker #AI
