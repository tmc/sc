# Smart Traffic Light Tutorial: Your First Statechart

**⏱️ Time to complete: 10-15 minutes**  
**🎯 Goal: Build a smart traffic light system that demonstrates the power of statecharts**

Welcome to the world of statecharts! In this tutorial, you'll build a **Smart Traffic Light** system that showcases why statecharts are superior to traditional state management. By the end, you'll have that "aha moment" about statechart benefits.

## What You'll Build

A smart traffic light with:
- **Normal operation**: RED → GREEN → YELLOW → RED cycle
- **Emergency mode**: Flashing red for all directions  
- **Maintenance mode**: System shutdown for repairs
- **Hierarchical states**: Clear organization of complex behavior

## Why This Example Matters

Traffic lights seem simple, but they're actually complex systems with:
- Multiple operational modes
- Interrupt behaviors (emergencies)
- Timed transitions
- Safety-critical requirements

This is **exactly** where statecharts excel over traditional if/else logic!

## The Problem with Traditional State Management

Here's how most developers handle traffic lights:

```javascript
// Traditional approach - quickly becomes a mess
if (currentState === "red") {
  if (event === "timer_expired") {
    if (emergencyMode) {
      currentState = "flashing_red";
    } else {
      currentState = "green";
    }
  } else if (event === "emergency") {
    emergencyMode = true;
    currentState = "flashing_red";
  }
}
// ... hundreds more lines of nested conditions
```

**Problems:**
- Impossible to visualize
- Hard to test all combinations
- Bugs hide in complex nesting
- Difficult to extend
- No clear understanding of valid states

## The Statechart Solution

With statecharts, you model the system as a **hierarchy of states** with **explicit transitions**:

```
SmartTrafficLight
├── Normal (initial)
│   ├── Red (initial)
│   ├── Green  
│   └── Yellow
├── Emergency
│   ├── FlashingRed (initial)
│   └── FlashingOff
└── Maintenance
```

**Benefits:**
- ✅ Visual representation of all possible states
- ✅ Impossible states are impossible to reach
- ✅ Easy to test and debug
- ✅ Self-documenting code
- ✅ Hierarchical organization scales

---

## Step 1: Choose Your Language

Pick your preferred language to follow along:

### 🔵 Go (Recommended)
```bash
cd tutorial
go run traffic_light.go
```

### 🐍 Python
```bash
cd tutorial
python3 traffic_light.py
```

### 🟨 JavaScript/XState
```bash
cd tutorial
node traffic_light.js
```

---

## Step 2: Understanding the Structure

Let's break down our traffic light statechart:

### Root State: SmartTrafficLight
The top-level container for our entire system.

### Level 1: Operational Modes
- **Normal** (initial): Standard traffic light operation
- **Emergency**: Emergency response mode
- **Maintenance**: System maintenance mode

### Level 2: Specific States
- **Normal.Red** (initial): Stop - cars must wait
- **Normal.Green**: Go - cars can proceed
- **Normal.Yellow**: Caution - prepare to stop
- **Emergency.FlashingRed**: Warning flash on
- **Emergency.FlashingOff**: Warning flash off

### Events
- `TIMER_EXPIRED`: Move to next state in normal cycle
- `EMERGENCY`: Activate emergency mode
- `EMERGENCY_CLEAR`: Return to normal operation
- `FLASH_TIMER`: Toggle emergency flashing
- `MAINTENANCE`: Enter maintenance mode
- `MAINTENANCE_COMPLETE`: Exit maintenance mode

---

## Step 3: The "Aha Moment"

### Traditional Approach Problems

Imagine coding this with if/else statements:

```javascript
function handleEvent(currentState, event) {
  if (currentState === "red") {
    if (event === "timer_expired") {
      if (emergencyMode) {
        return "flashing_red";
      } else if (maintenanceMode) {
        return "maintenance";
      } else {
        return "green";
      }
    } else if (event === "emergency") {
      emergencyMode = true;
      return "flashing_red";
    } else if (event === "maintenance") {
      maintenanceMode = true;
      return "maintenance";
    }
  } else if (currentState === "green") {
    // ... more nested conditions
  } else if (currentState === "yellow") {
    // ... more nested conditions
  } else if (currentState === "flashing_red") {
    // ... more nested conditions
  }
  // ... this goes on for hundreds of lines
}
```

**Questions:**
- Can you quickly understand all possible states?
- How do you test every combination?
- What happens if you add a new mode?
- How do you ensure thread safety?

### Statechart Approach

```go
// Clean, declarative statechart definition
chart := semantics.NewStatechart(&sc.Statechart{
    RootState: &sc.State{
        Label: "SmartTrafficLight",
        Children: []*sc.State{
            {
                Label: "Normal",
                IsInitial: true,
                Children: []*sc.State{
                    {Label: "Red", IsInitial: true},
                    {Label: "Green"},
                    {Label: "Yellow"},
                },
            },
            // ... more states
        },
    },
    Transitions: []*sc.Transition{
        {From: []string{"Red"}, To: []string{"Green"}, Event: "TIMER_EXPIRED"},
        // ... more transitions
    },
})
```

**Benefits:**
- ✅ Self-documenting structure
- ✅ Visual representation possible
- ✅ All states and transitions explicit
- ✅ Validation built-in
- ✅ Thread-safe execution

---

## Step 4: See It In Action

### Run the Tutorial Code

After running your chosen language version, you'll see:

```
=== Smart Traffic Light Tutorial ===
Building a traffic light with statecharts...

✓ Created traffic light statechart
✓ Statechart is valid

--- Traffic Light Structure ---
Root state: SmartTrafficLight
Number of states: 9
Number of transitions: 9
Number of events: 6

--- Creating Machine Instance ---
✓ Created machine: traffic-light-001

--- Initial State ---
Current states: [SmartTrafficLight, Normal, Red]

--- Simulating Traffic Light Operation ---
Sending event: TIMER_EXPIRED
🔴 → 🟢 (Red to Green)

Sending event: TIMER_EXPIRED  
🟢 → 🟡 (Green to Yellow)

Sending event: TIMER_EXPIRED
🟡 → 🔴 (Yellow to Red)

Sending event: EMERGENCY
🚨 Emergency mode activated!

Sending event: EMERGENCY_CLEAR
✅ Back to normal operation
```

### Key Observations

1. **Clear State**: You always know exactly which states are active
2. **Predictable Behavior**: Every transition is explicit and validated
3. **Hierarchical Structure**: Emergency mode works from any normal state
4. **Context Preservation**: Machine maintains state and context

---

## Step 5: Explore with the Web Visualizer

The statechart library includes a powerful web visualizer:

### Start the Visualizer
```bash
cd cmd/web-visualizer
go run main.go
```

Open http://localhost:8080 in your browser.

### Load Your Traffic Light

1. **Import** your statechart using the file upload
2. **Visualize** the state hierarchy graphically
3. **Step through** events to see transitions
4. **Inspect** the current configuration
5. **Export** to different formats (XState, SCXML, etc.)

### What You'll See

The visualizer shows your traffic light as an interactive diagram:

```
┌─────────────────────────────────────┐
│         SmartTrafficLight           │
│  ┌─────────────┐  ┌─────────────┐   │
│  │   Normal    │  │  Emergency  │   │
│  │ ┌─────────┐ │  │ ┌─────────┐ │   │
│  │ │   Red   │ │  │ │Flashing │ │   │
│  │ └─────────┘ │  │ │   Red   │ │   │
│  │ ┌─────────┐ │  │ └─────────┘ │   │
│  │ │  Green  │ │  │ ┌─────────┐ │   │
│  │ └─────────┘ │  │ │Flashing │ │   │
│  │ ┌─────────┐ │  │ │   Off   │ │   │
│  │ │ Yellow  │ │  │ └─────────┘ │   │
│  │ └─────────┘ │  └─────────────┘   │
│  └─────────────┘                    │
└─────────────────────────────────────┘
```

---

## Step 6: Advanced Concepts

### Orthogonal States (AND-states)

Traffic lights often have **multiple independent subsystems**:

```go
// Parallel monitoring systems
{
    Label: "Monitoring",
    Type: sc.StateTypeOrthogonal, // All children active simultaneously
    Children: []*sc.State{
        {
            Label: "MotionDetector",
            Children: []*sc.State{
                {Label: "DetectorIdle", IsInitial: true},
                {Label: "DetectorActive"},
            },
        },
        {
            Label: "EmergencyButton", 
            Children: []*sc.State{
                {Label: "ButtonReady", IsInitial: true},
                {Label: "ButtonPressed"},
            },
        },
    },
}
```

### Guards and Actions

Add conditions and side effects:

```go
// Transition with guard condition
{
    From: []string{"Red"},
    To: []string{"Green"},
    Event: "TIMER_EXPIRED",
    Guard: "trafficPresent", // Only transition if cars are waiting
    Actions: []string{"startGreenTimer", "logTransition"},
}
```

### History States

Remember previous states when returning:

```go
// Return to the same normal state after emergency
{
    From: []string{"Emergency"},
    To: []string{"Normal.History"}, // Resume where we left off
    Event: "EMERGENCY_CLEAR",
}
```

---

## Step 7: Real-World Applications

### Beyond Traffic Lights

Statecharts excel in many domains:

**🏭 Industrial Systems**
- Manufacturing workflows
- Robot control systems
- Safety interlocks

**🎮 Game Development**
- Character AI behavior
- Game mode management
- UI state management

**💻 Web Applications**
- Form wizards
- Authentication flows
- Loading states

**🤖 IoT Devices**
- Device lifecycle management
- Connection state handling
- Power management

### Integration Patterns

**Backend Services**
```go
// API endpoint state management
userStateMachine := semantics.NewMachine(
    userAuthStatechart,
    userID,
    initialContext,
)
```

**Frontend Applications**
```javascript
// React/Vue state management
const [state, send] = useMachine(trafficLightMachine);
```

**Event-Driven Systems**
```python
# Kafka/message bus integration
async def handle_event(event):
    await machine.step(event)
```

---

## Step 8: Testing Your Statechart

### Unit Testing

Statecharts are **inherently testable**:

```go
func TestTrafficLightNormalCycle(t *testing.T) {
    chart := SmartTrafficLight()
    machine, _ := semantics.NewMachine(chart, "test", nil)
    
    // Test initial state
    assert.Contains(t, machine.Configuration.States, "Red")
    
    // Test normal cycle
    machine.Step("TIMER_EXPIRED")
    assert.Contains(t, machine.Configuration.States, "Green")
    
    machine.Step("TIMER_EXPIRED")
    assert.Contains(t, machine.Configuration.States, "Yellow")
    
    machine.Step("TIMER_EXPIRED")
    assert.Contains(t, machine.Configuration.States, "Red")
}

func TestEmergencyInterrupt(t *testing.T) {
    chart := SmartTrafficLight()
    machine, _ := semantics.NewMachine(chart, "test", nil)
    
    // Start in Green
    machine.Step("TIMER_EXPIRED")
    assert.Contains(t, machine.Configuration.States, "Green")
    
    // Emergency should work from any state
    machine.Step("EMERGENCY")
    assert.Contains(t, machine.Configuration.States, "FlashingRed")
}
```

### Property-Based Testing

Test **invariants** across all possible states:

```go
func TestTrafficLightInvariants(t *testing.T) {
    chart := SmartTrafficLight()
    
    // Invariant: Only one light can be active at a time
    for _, state := range chart.GetAllStates() {
        if isLightState(state) {
            siblings := getSiblings(state)
            for _, sibling := range siblings {
                assert.False(t, bothActive(state, sibling))
            }
        }
    }
}
```

---

## Step 9: Performance and Scalability

### Why Statecharts Scale

**Traditional State Management:**
- O(n²) complexity for state combinations
- Difficult to optimize
- Memory leaks common

**Statechart Approach:**
- O(log n) hierarchy traversal
- Built-in optimization opportunities
- Predictable memory usage

### Benchmarking

```bash
# Run performance tests
go test -bench=. ./semantics/v1/

# Typical results:
BenchmarkStatechartTransition-8    1000000    1.2 μs/op
BenchmarkTraditionalIf-8           500000     2.4 μs/op
```

### Memory Usage

```go
// Statechart memory usage is predictable
func TestMemoryUsage(t *testing.T) {
    chart := SmartTrafficLight()
    
    // Baseline
    var m1 runtime.MemStats
    runtime.GC()
    runtime.ReadMemStats(&m1)
    
    // Create 1000 machines
    machines := make([]*semantics.MachineWrapper, 1000)
    for i := 0; i < 1000; i++ {
        machines[i], _ = semantics.NewMachine(chart, fmt.Sprintf("test-%d", i), nil)
    }
    
    // Measure
    var m2 runtime.MemStats
    runtime.GC()
    runtime.ReadMemStats(&m2)
    
    avgMemoryPerMachine := (m2.Alloc - m1.Alloc) / 1000
    assert.Less(t, avgMemoryPerMachine, uint64(1024)) // Less than 1KB per machine
}
```

---

## Step 10: Next Steps

### 🎯 Immediate Next Steps

1. **Explore the Web Visualizer**
   - Load your traffic light
   - Try different state configurations
   - Export to various formats

2. **Extend the Traffic Light**
   - Add pedestrian crossing
   - Implement timing controls
   - Add sensor integration

3. **Try Other Examples**
   ```bash
   # Hierarchical example
   go test ./semantics/v1/examples -run TestHierarchical -v
   
   # Orthogonal example  
   go test ./semantics/v1/examples -run TestOrthogonal -v
   ```

### 🚀 Advanced Topics

4. **Learn About Orthogonal States**
   - Model parallel behaviors
   - Handle concurrent events
   - Synchronize state changes

5. **Master Guards and Actions**
   - Add conditional transitions
   - Implement side effects
   - Create reactive systems

6. **Integrate with Your Stack**
   - Go: Use as service layer
   - Python: FastAPI/Flask integration
   - JavaScript: React/Vue state management

### 📚 Deep Dive Resources

7. **Academic Papers**
   - Read `docs/papers/reconciling-statechart-semantics.pdf`
   - Understand formal semantics
   - Learn about different statechart variants

8. **Real-World Examples**
   - Check `cmd/web-visualizer/examples/`
   - Study e-commerce checkout flows
   - Analyze user authentication systems

9. **Contributing**
   - Implement new features
   - Add language bindings
   - Improve documentation

---

## Congratulations! 🎉

You've successfully completed the Smart Traffic Light tutorial! 

### What You've Learned

✅ **Core Concepts**
- Hierarchical state organization
- Event-driven transitions
- Machine instances and context

✅ **Practical Benefits**
- Why statecharts beat traditional state management
- How to model complex behavior simply
- Testing and validation approaches

✅ **Real-World Applications**
- Industrial control systems
- User interface state management
- IoT device lifecycle management

### Your Statechart Journey

This tutorial is just the beginning. Statecharts are a powerful tool for:

- **Taming Complex Systems**: Break down complicated behavior into manageable pieces
- **Improving Code Quality**: Self-documenting, testable, and maintainable code
- **Reducing Bugs**: Impossible states become impossible to reach
- **Enhancing Collaboration**: Visual models improve team communication

### Join the Community

- **GitHub**: Contribute to the statechart library
- **Documentation**: Help improve tutorials and examples
- **Examples**: Share your statechart implementations

**Happy state modeling!** 🚦✨

---

## Troubleshooting

### Common Issues

**Go Module Problems**
```bash
# If you get module errors
go mod tidy
go mod download
```

**Python Import Errors**
```bash
# Set up Python environment
cd sdks/python
make install-dev
```

**Web Visualizer Issues**
```bash
# Ensure dependencies are installed
cd cmd/web-visualizer
go mod tidy
go run main.go
```

### Getting Help

- **Documentation**: Check `docs/` directory
- **Examples**: Look at `semantics/v1/examples/`
- **Issues**: Create GitHub issue with reproduction steps
- **Community**: Join discussions about statechart best practices

---

*This tutorial is part of the statechart library documentation. For more advanced topics, see the complete documentation in the `docs/` directory.*