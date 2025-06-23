"""
Flask integration for statecharts.

This module provides utilities for integrating statecharts with Flask applications,
including session-based state management and HTTP endpoint patterns.
"""

from typing import Optional, Dict, Any, Callable
from functools import wraps
import json

try:
    from flask import Flask, session, request, jsonify, g
except ImportError:
    Flask = None
    session = None
    request = None
    jsonify = None
    g = None

from ..core import Statechart, Machine, Event
from ..factory import machine


class StatechartSession:
    """
    Flask session-based statechart manager.
    
    This class manages statechart machines using Flask's session storage,
    allowing stateful interactions across HTTP requests.
    """
    
    def __init__(self, session_key: str = "statechart_machines"):
        """
        Initialize the session manager.
        
        Args:
            session_key: Key to use for storing machines in the session
        """
        if Flask is None:
            raise ImportError("Flask is required for Flask integration")
        
        self.session_key = session_key
    
    def get_machine(self, machine_id: str) -> Optional[Machine]:
        """
        Get a machine from the session.
        
        Args:
            machine_id: ID of the machine to retrieve
            
        Returns:
            The machine if found, None otherwise
        """
        machines_data = session.get(self.session_key, {})
        
        if machine_id in machines_data:
            # Deserialize machine from session data
            machine_dict = machines_data[machine_id]
            # Note: In a real implementation, you'd want to serialize/deserialize
            # the machine properly. This is a simplified version.
            return machine_dict
        
        return None
    
    def save_machine(self, machine: Machine) -> None:
        """
        Save a machine to the session.
        
        Args:
            machine: The machine to save
        """
        if self.session_key not in session:
            session[self.session_key] = {}
        
        # Note: In a real implementation, you'd want to serialize the machine
        # to a format that can be stored in the session
        session[self.session_key][machine.id] = machine.to_dict()
        session.modified = True
    
    def create_machine(self, machine_id: str, statechart: Statechart, context: Optional[Dict[str, Any]] = None) -> Machine:
        """
        Create a new machine and save it to the session.
        
        Args:
            machine_id: ID for the new machine
            statechart: The statechart definition
            context: Optional initial context
            
        Returns:
            The created machine
        """
        new_machine = machine(machine_id, statechart, context)
        self.save_machine(new_machine)
        return new_machine
    
    def delete_machine(self, machine_id: str) -> bool:
        """
        Delete a machine from the session.
        
        Args:
            machine_id: ID of the machine to delete
            
        Returns:
            True if the machine was deleted, False if not found
        """
        machines_data = session.get(self.session_key, {})
        
        if machine_id in machines_data:
            del machines_data[machine_id]
            session[self.session_key] = machines_data
            session.modified = True
            return True
        
        return False


class FlaskStatechartBlueprint:
    """
    Flask blueprint for statechart HTTP endpoints.
    
    This class provides a blueprint with common endpoints for managing
    statechart machines via HTTP.
    """
    
    def __init__(self, name: str = "statecharts", url_prefix: str = "/statecharts"):
        """
        Initialize the blueprint.
        
        Args:
            name: Name of the blueprint
            url_prefix: URL prefix for all routes
        """
        if Flask is None:
            raise ImportError("Flask is required for Flask integration")
        
        from flask import Blueprint
        
        self.blueprint = Blueprint(name, __name__, url_prefix=url_prefix)
        self.session_manager = StatechartSession()
        self.statechart_registry: Dict[str, Statechart] = {}
        
        self._register_routes()
    
    def register_statechart(self, statechart_id: str, statechart: Statechart) -> None:
        """
        Register a statechart definition.
        
        Args:
            statechart_id: ID for the statechart
            statechart: The statechart definition
        """
        self.statechart_registry[statechart_id] = statechart
    
    def _register_routes(self):
        """Register the HTTP routes."""
        
        @self.blueprint.route("/machines", methods=["POST"])
        def create_machine():
            """Create a new machine instance."""
            data = request.get_json()
            
            if not data or "statechart_id" not in data:
                return jsonify({"error": "statechart_id is required"}), 400
            
            statechart_id = data["statechart_id"]
            machine_id = data.get("machine_id", f"machine-{statechart_id}")
            context = data.get("context", {})
            
            if statechart_id not in self.statechart_registry:
                return jsonify({"error": f"Statechart '{statechart_id}' not found"}), 404
            
            statechart = self.statechart_registry[statechart_id]
            new_machine = self.session_manager.create_machine(machine_id, statechart, context)
            
            return jsonify({
                "machine_id": new_machine.id,
                "state": new_machine.state.name,
                "active_states": new_machine.get_active_states(),
                "context": new_machine.context
            })
        
        @self.blueprint.route("/machines/<machine_id>", methods=["GET"])
        def get_machine(machine_id: str):
            """Get machine status."""
            machine = self.session_manager.get_machine(machine_id)
            
            if not machine:
                return jsonify({"error": f"Machine '{machine_id}' not found"}), 404
            
            return jsonify({
                "machine_id": machine_id,
                "state": machine.get("state", "UNKNOWN"),
                "active_states": machine.get("active_states", []),
                "context": machine.get("context", {}),
                "step_count": machine.get("step_count", 0)
            })
        
        @self.blueprint.route("/machines/<machine_id>/events", methods=["POST"])
        def send_event(machine_id: str):
            """Send an event to a machine."""
            data = request.get_json()
            
            if not data or "event" not in data:
                return jsonify({"error": "event is required"}), 400
            
            machine = self.session_manager.get_machine(machine_id)
            if not machine:
                return jsonify({"error": f"Machine '{machine_id}' not found"}), 404
            
            event_label = data["event"]
            event_context = data.get("context", {})
            
            # Note: In a real implementation, you would process the event
            # and update the machine state accordingly
            
            return jsonify({
                "machine_id": machine_id,
                "event": event_label,
                "status": "processed",
                "new_state": machine.get("state", "UNKNOWN"),
                "active_states": machine.get("active_states", [])
            })
        
        @self.blueprint.route("/machines/<machine_id>", methods=["DELETE"])
        def delete_machine(machine_id: str):
            """Delete a machine."""
            deleted = self.session_manager.delete_machine(machine_id)
            
            if not deleted:
                return jsonify({"error": f"Machine '{machine_id}' not found"}), 404
            
            return jsonify({"message": f"Machine '{machine_id}' deleted"})


def require_machine(machine_id_param: str = "machine_id"):
    """
    Decorator to require a valid machine in the session.
    
    Args:
        machine_id_param: Name of the parameter containing the machine ID
        
    Returns:
        Decorator function
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if Flask is None:
                raise ImportError("Flask is required for Flask integration")
            
            machine_id = kwargs.get(machine_id_param) or request.view_args.get(machine_id_param)
            
            if not machine_id:
                return jsonify({"error": f"Missing {machine_id_param}"}), 400
            
            session_manager = StatechartSession()
            machine = session_manager.get_machine(machine_id)
            
            if not machine:
                return jsonify({"error": f"Machine '{machine_id}' not found"}), 404
            
            g.machine = machine
            g.machine_id = machine_id
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


# Example usage patterns

def create_workflow_app():
    """
    Example: Create a Flask app with a simple workflow statechart.
    
    This demonstrates how to integrate statecharts into a Flask application
    for managing user workflows.
    """
    if Flask is None:
        return None
    
    app = Flask(__name__)
    app.secret_key = "your-secret-key-here"  # Change this in production!
    
    # Create a simple workflow statechart
    from ..factory import basic_state, normal_state, transition, event, statechart
    
    # Define workflow states
    start = basic_state("Start", is_initial=True)
    in_progress = basic_state("InProgress")
    review = basic_state("Review")
    completed = basic_state("Completed", is_final=True)
    
    workflow_root = normal_state("Workflow", [start, in_progress, review, completed])
    
    # Define transitions
    workflow_transitions = [
        transition("Begin", "Start", "InProgress", "BEGIN"),
        transition("Submit", "InProgress", "Review", "SUBMIT"),
        transition("Approve", "Review", "Completed", "APPROVE"),
        transition("Reject", "Review", "InProgress", "REJECT"),
    ]
    
    # Define events
    workflow_events = [
        event("BEGIN"),
        event("SUBMIT"),
        event("APPROVE"),
        event("REJECT"),
    ]
    
    workflow_chart = statechart(workflow_root, workflow_transitions, workflow_events)
    
    # Create and register blueprint
    statechart_bp = FlaskStatechartBlueprint()
    statechart_bp.register_statechart("workflow", workflow_chart)
    app.register_blueprint(statechart_bp.blueprint)
    
    # Add a simple frontend route
    @app.route("/")
    def index():
        return '''
        <!DOCTYPE html>
        <html>
        <head><title>Workflow Demo</title></head>
        <body>
            <h1>Statechart Workflow Demo</h1>
            <div id="status"></div>
            <button onclick="createWorkflow()">Create Workflow</button>
            <button onclick="sendEvent('BEGIN')">Begin</button>
            <button onclick="sendEvent('SUBMIT')">Submit</button>
            <button onclick="sendEvent('APPROVE')">Approve</button>
            <button onclick="sendEvent('REJECT')">Reject</button>
            
            <script>
                let currentMachineId = null;
                
                async function createWorkflow() {
                    const response = await fetch('/statecharts/machines', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            statechart_id: 'workflow',
                            machine_id: 'workflow-' + Date.now()
                        })
                    });
                    const data = await response.json();
                    currentMachineId = data.machine_id;
                    updateStatus();
                }
                
                async function sendEvent(eventName) {
                    if (!currentMachineId) {
                        alert('Create a workflow first');
                        return;
                    }
                    
                    const response = await fetch(`/statecharts/machines/${currentMachineId}/events`, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({event: eventName})
                    });
                    updateStatus();
                }
                
                async function updateStatus() {
                    if (!currentMachineId) return;
                    
                    const response = await fetch(`/statecharts/machines/${currentMachineId}`);
                    const data = await response.json();
                    document.getElementById('status').innerHTML = 
                        `<h3>Machine: ${data.machine_id}</h3>
                         <p>State: ${data.state}</p>
                         <p>Active States: ${data.active_states.join(', ')}</p>`;
                }
            </script>
        </body>
        </html>
        '''
    
    return app


# Utility functions for common patterns

def machine_state_middleware(app: Flask, session_manager: Optional[StatechartSession] = None):
    """
    Add middleware to automatically load machine state into Flask's g object.
    
    Args:
        app: Flask application instance
        session_manager: Optional session manager (creates default if None)
    """
    if session_manager is None:
        session_manager = StatechartSession()
    
    @app.before_request
    def load_machine_state():
        machine_id = request.view_args.get("machine_id") if request.view_args else None
        if machine_id:
            g.machine = session_manager.get_machine(machine_id)
            g.machine_id = machine_id
    
    return session_manager


def statechart_error_handler(app: Flask):
    """
    Add error handlers for common statechart-related errors.
    
    Args:
        app: Flask application instance
    """
    @app.errorhandler(404)
    def machine_not_found(error):
        if request.path.startswith("/statecharts/"):
            return jsonify({"error": "Machine not found"}), 404
        return error
    
    @app.errorhandler(400)
    def bad_request(error):
        if request.path.startswith("/statecharts/"):
            return jsonify({"error": "Bad request"}), 400
        return error