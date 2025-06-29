/**
 * Smart Traffic Light Tutorial - JavaScript/XState Version
 * 
 * This tutorial demonstrates how to build a smart traffic light system using
 * XState (JavaScript) that can be converted to the statechart library format.
 * 
 * The traffic light showcases:
 * 1. Hierarchical state composition (Normal vs Emergency modes)
 * 2. State transitions with events
 * 3. Emergency interrupt behavior  
 * 4. Machine instances with context
 * 
 * This is a "Hello World" example showing the practical power of statecharts
 * for modeling complex behavior in a simple, understandable way.
 */


// XState machine definition for smart traffic light
const smartTrafficLightMachine = {
  id: 'smartTrafficLight',
  initial: 'normal',
  context: {
    timer: 0,
    emergencyMode: false,
    lastMaintenance: '2024-01-01'
  },
  states: {
    normal: {
      initial: 'red',
      states: {
        red: {
          on: {
            TIMER_EXPIRED: 'green'
          }
        },
        green: {
          on: {
            TIMER_EXPIRED: 'yellow'
          }
        },
        yellow: {
          on: {
            TIMER_EXPIRED: 'red'
          }
        }
      },
      on: {
        EMERGENCY: 'emergency',
        MAINTENANCE: 'maintenance'
      }
    },
    emergency: {
      initial: 'flashingRed',
      states: {
        flashingRed: {
          on: {
            FLASH_TIMER: 'flashingOff'
          }
        },
        flashingOff: {
          on: {
            FLASH_TIMER: 'flashingRed'
          }
        }
      },
      on: {
        EMERGENCY_CLEAR: 'normal',
        MAINTENANCE: 'maintenance'
      }
    },
    maintenance: {
      on: {
        MAINTENANCE_COMPLETE: 'normal'
      }
    }
  }
};

// Tutorial demonstration functions
function demonstrateTrafficLight() {
  console.log('=== Smart Traffic Light Tutorial - JavaScript ===');
  console.log('Building a traffic light with statecharts...');
  console.log('');

  // Step 1: Show the machine definition
  console.log('✓ Created traffic light statechart');
  
  // Step 2: Analyze structure
  console.log('\n--- Traffic Light Structure ---');
  console.log(`Root state: ${smartTrafficLightMachine.id}`);
  console.log(`Initial state: ${smartTrafficLightMachine.initial}`);
  
  // Count states recursively
  const stateCount = countStates(smartTrafficLightMachine.states);
  console.log(`Number of states: ${stateCount}`);
  
  // Count transitions
  const transitionCount = countTransitions(smartTrafficLightMachine.states);
  console.log(`Number of transitions: ${transitionCount}`);
  
  // Step 3: Show all states
  console.log('\nAll states:');
  printStates(smartTrafficLightMachine.states, '');
  
  // Step 4: Show transitions
  console.log('\nTransitions:');
  printTransitions(smartTrafficLightMachine.states, '');
  
  // Step 5: Simulate machine execution
  console.log('\n--- Simulating Traffic Light Operation ---');
  simulateTrafficLight();
  
  // Step 6: Show the power of statecharts
  console.log('\n--- Why Statecharts Are Powerful ---');
  console.log('Traditional if/else approach for traffic lights:');
  console.log(`
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
// ... many more nested conditions
  `);
  
  console.log('With statecharts:');
  console.log('- Clear visual representation of all possible states');
  console.log('- Hierarchical organization (Normal/Emergency modes)'); 
  console.log('- Explicit transitions with events');
  console.log('- Impossible states are impossible to reach');
  console.log('- Easy to test, debug, and extend');
  
  console.log('\n--- Conversion to Statechart Library Format ---');
  console.log('This XState machine can be converted to the statechart library format:');
  
  const convertedFormat = convertToStatechartFormat(smartTrafficLightMachine);
  console.log(JSON.stringify(convertedFormat, null, 2));
  
  console.log('\n--- Tutorial Complete! ---');
  console.log('You\'ve successfully:');
  console.log('✓ Created a hierarchical statechart in JavaScript');
  console.log('✓ Modeled complex behavior with clear states');
  console.log('✓ Demonstrated emergency interrupt patterns');
  console.log('✓ Seen conversion to statechart library format');
  console.log('✓ Understood why statecharts beat traditional state management');
  
  console.log('\nNext steps:');
  console.log('- Try the web visualizer to see your statechart graphically');
  console.log('- Explore parallel states for concurrent behavior');
  console.log('- Add guards and actions for complex logic');
  console.log('- Use the bridge to convert between XState and the library');
  console.log('- Check out the Go and Python versions');
}

// Helper functions
function countStates(states, count = 0) {
  for (const stateName in states) {
    count++;
    const state = states[stateName];
    if (state.states) {
      count = countStates(state.states, count);
    }
  }
  return count;
}

function countTransitions(states, count = 0) {
  for (const stateName in states) {
    const state = states[stateName];
    if (state.on) {
      count += Object.keys(state.on).length;
    }
    if (state.states) {
      count = countTransitions(state.states, count);
    }
  }
  return count;
}

function printStates(states, indent) {
  for (const stateName in states) {
    const state = states[stateName];
    const initial = state.initial ? ` (initial: ${state.initial})` : '';
    console.log(`${indent}  - ${stateName}${initial}`);
    if (state.states) {
      printStates(state.states, indent + '  ');
    }
  }
}

function printTransitions(states, path) {
  for (const stateName in states) {
    const state = states[stateName];
    const fullPath = path ? `${path}.${stateName}` : stateName;
    
    if (state.on) {
      for (const event in state.on) {
        const target = state.on[event];
        console.log(`  - ${fullPath} --${event}--> ${target}`);
      }
    }
    
    if (state.states) {
      printTransitions(state.states, fullPath);
    }
  }
}

function simulateTrafficLight() {
  let currentState = 'normal.red';
  const context = { ...smartTrafficLightMachine.context };
  
  console.log(`Initial state: ${currentState}`);
  
  // Simulate normal operation
  const events = ['TIMER_EXPIRED', 'TIMER_EXPIRED', 'TIMER_EXPIRED'];
  events.forEach(event => {
    console.log(`\nSending event: ${event}`);
    // Simplified state transition logic
    if (currentState === 'normal.red' && event === 'TIMER_EXPIRED') {
      currentState = 'normal.green';
      console.log(`State changed to: ${currentState} (🟢 GREEN LIGHT)`);
    } else if (currentState === 'normal.green' && event === 'TIMER_EXPIRED') {
      currentState = 'normal.yellow';
      console.log(`State changed to: ${currentState} (🟡 YELLOW LIGHT)`);
    } else if (currentState === 'normal.yellow' && event === 'TIMER_EXPIRED') {
      currentState = 'normal.red';
      console.log(`State changed to: ${currentState} (🔴 RED LIGHT)`);
    }
  });
  
  // Simulate emergency
  console.log(`\nSending event: EMERGENCY`);
  currentState = 'emergency.flashingRed';
  context.emergencyMode = true;
  console.log(`State changed to: ${currentState} (🚨 EMERGENCY MODE)`);
  
  console.log(`\nSending event: EMERGENCY_CLEAR`);
  currentState = 'normal.red';
  context.emergencyMode = false;
  console.log(`State changed to: ${currentState} (Back to normal operation)`);
}

function convertToStatechartFormat(xstateMachine) {
  // Simplified conversion to demonstrate the concept
  return {
    rootState: {
      label: xstateMachine.id,
      type: 'NORMAL',
      children: Object.keys(xstateMachine.states).map(stateName => ({
        label: stateName,
        type: xstateMachine.states[stateName].states ? 'NORMAL' : 'BASIC',
        isInitial: stateName === xstateMachine.initial
      }))
    },
    events: extractEvents(xstateMachine.states),
    transitions: extractTransitions(xstateMachine.states)
  };
}

function extractEvents(states) {
  const events = new Set();
  
  function traverse(stateObj) {
    for (const stateName in stateObj) {
      const state = stateObj[stateName];
      if (state.on) {
        Object.keys(state.on).forEach(event => events.add(event));
      }
      if (state.states) {
        traverse(state.states);
      }
    }
  }
  
  traverse(states);
  return Array.from(events).map(event => ({ label: event }));
}

function extractTransitions(states) {
  const transitions = [];
  
  function traverse(stateObj, parentPath = '') {
    for (const stateName in stateObj) {
      const state = stateObj[stateName];
      const fullPath = parentPath ? `${parentPath}.${stateName}` : stateName;
      
      if (state.on) {
        for (const event in state.on) {
          const target = state.on[event];
          transitions.push({
            label: `${stateName}To${target}`,
            from: [stateName],
            to: [target],
            event: event
          });
        }
      }
      
      if (state.states) {
        traverse(state.states, fullPath);
      }
    }
  }
  
  traverse(states);
  return transitions;
}

// Run the tutorial if this is the main module
if (typeof module !== 'undefined' && require.main === module) {
  demonstrateTrafficLight();
} else if (typeof window !== 'undefined') {
  // Browser environment
  window.demonstrateTrafficLight = demonstrateTrafficLight;
  window.smartTrafficLightMachine = smartTrafficLightMachine;
}

// Export for use in other modules
if (typeof module !== 'undefined') {
  module.exports = {
    smartTrafficLightMachine,
    demonstrateTrafficLight
  };
}