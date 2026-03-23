from graph.analyst_graph import graph


# Ask user question
question = input("Enter your business question: ")

# Choose execution mode
mode = input("Choose mode (autonomous / guided / collaborative): ")


state = {
    "business_question": question,
    "dataset_path": "data/marketing.csv",
    "mode": mode
}


result = graph.invoke(state)

print("\n===== FINAL REPORT =====\n")

print(result.get("final_report", "No report generated"))

'''from tools.load_data import load_csv
from tools.dataset_profiler import profile_dataset
from tools.data_validation import validate_data
from tools.eda_tools import eda_summary
from tools.stats_tools import correlation_test
from agents.analyst_agent import generate_insights
from graph.analyst_graph import graph # <- this line needs checking

question = "Is advertising spend correlated with revenue?"

df = load_csv("data/marketing.csv")

profile = profile_dataset(df)

validation = validate_data(df)

eda = eda_summary(df)

stats = correlation_test(df, "ad_spend", "revenue")

state = {

    "business_question": "Why did revenue change this quarter?",
    "dataset_path": "data/sales.csv"

}

result = graph.invoke(state) # <- the mentioned line above works with this part so check it too 

print(result["final_report"])

insights = generate_insights(
    question,
    profile,
    validation,
    eda,
    stats
)

print(insights)'''