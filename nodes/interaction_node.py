from state.state import AnalystState


def interaction_node(state: AnalystState) -> AnalystState:

    mode = state.get("mode", "autonomous")
    plan = state.get("analysis_plan")
    if not plan:
        plan = state.get("analysis_evidence", {}).get("analysis_plan", [])

    # AUTONOMOUS MODE
    if mode == "autonomous":
        print("\n[Agent] Running in autonomous mode.")
        return state

    # GUIDED MODE
    if mode == "guided":

        print("\n[Agent] Proposed Analysis Plan:")

        for step in plan:
            print(f" - {step}")

        user_input = input(
            "\nApprove plan? (yes / modify / stop): "
        ).strip().lower()

        if user_input == "yes":
            return state

        elif user_input == "modify":

            new_plan = input(
                "Enter new analysis steps separated by commas:\n"
            )

            new_plan_list = [
                step.strip() for step in new_plan.split(",")
            ]

            state["analysis_plan"] = new_plan_list
            state.setdefault("analysis_evidence", {})["analysis_plan"] = new_plan_list

            print("\n[Agent] Updated plan:")
            for step in new_plan_list:
                print(f" - {step}")

            return state

        elif user_input == "stop":

            print("\n[Agent] Execution stopped by user.")
            state["awaiting_user"] = True
            return state

    '''if mode == "guided":

        plan = state.get("analysis_plan", [])

        print("\nAgent proposes the following analyses:")
        for step in plan:
            print("-", step)

        user = input("\nApprove plan? (yes / modify): ")

        if user.lower() != "yes":
            new_plan = input("Enter analyses separated by commas: ")
            state["analysis_plan"] = [x.strip() for x in new_plan.split(",")]'''

    if mode == "collaborative":

        question = input("\nHow would you like to proceed with the analysis? ")
        state["user_response"] = question

    return state
