load("sc", "sc")

def traffic_light():
    return sc.machine(
        name = "traffic_light",
        states = [
            sc.state("green", initial=True, transitions=[
                sc.transition(to="yellow", event="TIMER"),
            ]),
            sc.state("yellow", transitions=[
                sc.transition(to="red", event="TIMER"),
            ]),
            sc.state("red", transitions=[
                sc.transition(to="green", event="TIMER"),
            ]),
        ],
    )

def main():
    m = traffic_light()
    print("Machine created:", m)
    inst = m.start()
    print("Machine started. Current config:", inst.matches("green"))
    inst.send("TIMER")
    print("Transitioned. Current config:", inst.matches("yellow"))
    inst.send("TIMER") 
    print("Transitioned. Current config:", inst.matches("red"))
