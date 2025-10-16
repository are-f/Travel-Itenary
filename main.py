from agent import agent_run

def main():
    yourbudget = input("Enter your budget: ")
    yourlocation = input("Enter your location: ")
    noofdays = int(input("Enter the number of days: "))
    yourquery = f"""Suggest me three travel itenaries around {yourlocation} for {noofdays} days. I have a budget of AED {yourbudget}."""

    print(agent_run(query=yourquery))

if __name__ == "__main__":
    main()
