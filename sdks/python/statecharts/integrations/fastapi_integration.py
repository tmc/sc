"""
FastAPI integration for statecharts.

This module provides utilities for integrating statecharts with FastAPI applications,
including dependency injection, WebSocket support, and async operation patterns.
"""

from typing import Optional, Dict, Any, List, Callable, AsyncGenerator
from contextlib import asynccontextmanager
import json
import asyncio
from dataclasses import dataclass

try:
    from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    FastAPI = None
    HTTPException = None
    Depends = None
    WebSocket = None
    WebSocketDisconnect = None
    JSONResponse = None
    BaseModel = None

from ..core import Statechart, Machine, Event, StateType
from ..factory import machine


# Pydantic models for API

class CreateMachineRequest(BaseModel):
    """Request model for creating a machine."""
    statechart_id: str
    machine_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class MachineResponse(BaseModel):
    """Response model for machine information."""
    machine_id: str
    state: str
    active_states: List[str]
    context: Dict[str, Any]
    step_count: int


class SendEventRequest(BaseModel):
    """Request model for sending events."""
    event: str
    context: Optional[Dict[str, Any]] = None


class EventResponse(BaseModel):
    """Response model for event processing."""
    machine_id: str
    event: str
    status: str
    new_state: str
    active_states: List[str]


@dataclass
class StatechartRegistry:
    """Registry for managing statechart definitions."""
    
    def __init__(self):
        self.statecharts: Dict[str, Statechart] = {}
        self.machines: Dict[str, Machine] = {}
    
    def register_statechart(self, statechart_id: str, statechart: Statechart) -> None:
        """Register a statechart definition."""
        self.statecharts[statechart_id] = statechart
    
    def get_statechart(self, statechart_id: str) -> Optional[Statechart]:
        """Get a statechart by ID."""
        return self.statecharts.get(statechart_id)
    
    def create_machine(self, statechart_id: str, machine_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Machine:
        """Create a new machine instance."""
        statechart = self.get_statechart(statechart_id)
        if not statechart:
            raise ValueError(f"Statechart '{statechart_id}' not found")
        
        if machine_id is None:
            machine_id = f"machine-{statechart_id}-{len(self.machines)}"
        
        new_machine = machine(machine_id, statechart, context)
        self.machines[machine_id] = new_machine
        return new_machine
    
    def get_machine(self, machine_id: str) -> Optional[Machine]:
        """Get a machine by ID."""
        return self.machines.get(machine_id)
    
    def delete_machine(self, machine_id: str) -> bool:
        """Delete a machine."""
        if machine_id in self.machines:
            del self.machines[machine_id]
            return True
        return False
    
    def list_machines(self) -> List[str]:
        """List all machine IDs."""
        return list(self.machines.keys())


class StatechartService:
    """Service for managing statechart operations."""
    
    def __init__(self):
        self.registry = StatechartRegistry()
        self.event_listeners: Dict[str, List[Callable]] = {}
    
    async def create_machine(self, request: CreateMachineRequest) -> Machine:
        """Create a new machine."""
        try:
            return self.registry.create_machine(
                request.statechart_id,
                request.machine_id,
                request.context
            )
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    
    async def get_machine(self, machine_id: str) -> Machine:
        """Get a machine by ID."""
        machine = self.registry.get_machine(machine_id)
        if not machine:
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")
        return machine
    
    async def send_event(self, machine_id: str, request: SendEventRequest) -> EventResponse:
        """Send an event to a machine."""
        machine = await self.get_machine(machine_id)
        
        # Note: In a real implementation, you would process the event
        # and update the machine state accordingly
        
        # Notify event listeners
        await self._notify_listeners(machine_id, request.event, request.context)
        
        return EventResponse(
            machine_id=machine_id,
            event=request.event,
            status="processed",
            new_state=machine.state.name,
            active_states=machine.get_active_states()
        )
    
    async def delete_machine(self, machine_id: str) -> bool:
        """Delete a machine."""
        return self.registry.delete_machine(machine_id)
    
    def add_event_listener(self, machine_id: str, listener: Callable):
        """Add an event listener for a machine."""
        if machine_id not in self.event_listeners:
            self.event_listeners[machine_id] = []
        self.event_listeners[machine_id].append(listener)
    
    async def _notify_listeners(self, machine_id: str, event: str, context: Optional[Dict[str, Any]]):
        """Notify event listeners."""
        if machine_id in self.event_listeners:
            for listener in self.event_listeners[machine_id]:
                if asyncio.iscoroutinefunction(listener):
                    await listener(machine_id, event, context)
                else:
                    listener(machine_id, event, context)


# Dependency injection

def get_statechart_service() -> StatechartService:
    """Dependency injection for StatechartService."""
    # In a real application, you might want to use a singleton or manage this differently
    if not hasattr(get_statechart_service, '_service'):
        get_statechart_service._service = StatechartService()
    return get_statechart_service._service


def get_machine(machine_id: str, service: StatechartService = Depends(get_statechart_service)) -> Machine:
    """Dependency injection for getting a machine."""
    machine = service.registry.get_machine(machine_id)
    if not machine:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")
    return machine


# FastAPI router factory

def create_statechart_router(service: Optional[StatechartService] = None):
    """
    Create a FastAPI router with statechart endpoints.
    
    Args:
        service: Optional service instance (creates default if None)
        
    Returns:
        FastAPI router with statechart endpoints
    """
    if FastAPI is None:
        raise ImportError("FastAPI is required for FastAPI integration")
    
    from fastapi import APIRouter
    
    router = APIRouter(prefix="/statecharts", tags=["statecharts"])
    
    if service is None:
        service = StatechartService()
    
    @router.post("/machines", response_model=MachineResponse)
    async def create_machine(request: CreateMachineRequest):
        """Create a new machine instance."""
        machine = await service.create_machine(request)
        return MachineResponse(
            machine_id=machine.id,
            state=machine.state.name,
            active_states=machine.get_active_states(),
            context=machine.context,
            step_count=len(machine.step_history)
        )
    
    @router.get("/machines/{machine_id}", response_model=MachineResponse)
    async def get_machine(machine_id: str):
        """Get machine status."""
        machine = await service.get_machine(machine_id)
        return MachineResponse(
            machine_id=machine.id,
            state=machine.state.name,
            active_states=machine.get_active_states(),
            context=machine.context,
            step_count=len(machine.step_history)
        )
    
    @router.post("/machines/{machine_id}/events", response_model=EventResponse)
    async def send_event(machine_id: str, request: SendEventRequest):
        """Send an event to a machine."""
        return await service.send_event(machine_id, request)
    
    @router.delete("/machines/{machine_id}")
    async def delete_machine(machine_id: str):
        """Delete a machine."""
        deleted = await service.delete_machine(machine_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")
        return {"message": f"Machine '{machine_id}' deleted"}
    
    @router.get("/machines")
    async def list_machines():
        """List all machines."""
        machines = service.registry.list_machines()
        return {"machines": machines}
    
    return router


# WebSocket support

class WebSocketManager:
    """Manager for WebSocket connections to machines."""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.service = StatechartService()
    
    async def connect(self, websocket: WebSocket, machine_id: str):
        """Connect a WebSocket to a machine."""
        await websocket.accept()
        
        if machine_id not in self.active_connections:
            self.active_connections[machine_id] = []
        
        self.active_connections[machine_id].append(websocket)
        
        # Add event listener for this machine
        self.service.add_event_listener(machine_id, self._websocket_listener)
    
    def disconnect(self, websocket: WebSocket, machine_id: str):
        """Disconnect a WebSocket from a machine."""
        if machine_id in self.active_connections:
            if websocket in self.active_connections[machine_id]:
                self.active_connections[machine_id].remove(websocket)
            
            if not self.active_connections[machine_id]:
                del self.active_connections[machine_id]
    
    async def send_to_machine(self, machine_id: str, message: dict):
        """Send a message to all WebSockets connected to a machine."""
        if machine_id in self.active_connections:
            for connection in self.active_connections[machine_id]:
                try:
                    await connection.send_json(message)
                except:
                    # Connection might be closed
                    pass
    
    async def _websocket_listener(self, machine_id: str, event: str, context: Optional[Dict[str, Any]]):
        """WebSocket event listener."""
        message = {
            "type": "event",
            "machine_id": machine_id,
            "event": event,
            "context": context
        }
        await self.send_to_machine(machine_id, message)


def create_websocket_endpoint(manager: WebSocketManager):
    """Create WebSocket endpoint for real-time machine updates."""
    
    async def websocket_endpoint(websocket: WebSocket, machine_id: str):
        await manager.connect(websocket, machine_id)
        try:
            while True:
                # Keep connection alive and handle incoming messages
                data = await websocket.receive_json()
                
                if data.get("type") == "send_event":
                    # Handle event sending via WebSocket
                    event_request = SendEventRequest(
                        event=data["event"],
                        context=data.get("context")
                    )
                    await manager.service.send_event(machine_id, event_request)
                
        except WebSocketDisconnect:
            manager.disconnect(websocket, machine_id)
    
    return websocket_endpoint


# Example application factory

def create_statechart_app():
    """
    Create a complete FastAPI application with statechart support.
    
    Returns:
        FastAPI application instance
    """
    if FastAPI is None:
        return None
    
    app = FastAPI(title="Statechart API", version="1.0.0")
    
    # Create service and router
    service = StatechartService()
    router = create_statechart_router(service)
    app.include_router(router)
    
    # WebSocket support
    websocket_manager = WebSocketManager()
    websocket_endpoint = create_websocket_endpoint(websocket_manager)
    app.add_websocket_route("/ws/{machine_id}", websocket_endpoint)
    
    # Example statechart registration
    @app.on_event("startup")
    async def register_example_statecharts():
        from ..factory import basic_state, normal_state, transition, event, statechart
        
        # Simple toggle statechart
        off = basic_state("Off", is_initial=True)
        on = basic_state("On")
        root = normal_state("Toggle", [off, on])
        
        transitions = [
            transition("TurnOn", "Off", "On", "TOGGLE"),
            transition("TurnOff", "On", "Off", "TOGGLE"),
        ]
        
        events = [event("TOGGLE")]
        
        toggle_chart = statechart(root, transitions, events)
        service.registry.register_statechart("toggle", toggle_chart)
    
    # Add CORS middleware for web frontend
    try:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except ImportError:
        pass
    
    return app


# Async context manager for machine lifecycle

@asynccontextmanager
async def managed_machine(service: StatechartService, statechart_id: str, context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Machine, None]:
    """
    Async context manager for machine lifecycle management.
    
    Args:
        service: StatechartService instance
        statechart_id: ID of the statechart to instantiate
        context: Optional initial context
        
    Yields:
        Machine instance
    """
    request = CreateMachineRequest(statechart_id=statechart_id, context=context)
    machine = await service.create_machine(request)
    
    try:
        yield machine
    finally:
        await service.delete_machine(machine.id)


# Utility decorators

def require_machine_state(*required_states: str):
    """
    Decorator to require machine to be in specific states.
    
    Args:
        required_states: State labels that the machine must be in
        
    Returns:
        Decorator function
    """
    def decorator(func):
        async def wrapper(machine_id: str, *args, **kwargs):
            service = get_statechart_service()
            machine = await service.get_machine(machine_id)
            
            active_states = machine.get_active_states()
            if not any(state in active_states for state in required_states):
                raise HTTPException(
                    status_code=400,
                    detail=f"Machine must be in one of states: {required_states}"
                )
            
            return await func(machine_id, *args, **kwargs)
        return wrapper
    return decorator


# Example usage patterns

async def example_usage():
    """Example of using the FastAPI integration."""
    service = StatechartService()
    
    # Register a statechart
    from ..factory import simple_toggle_statechart
    toggle_chart = simple_toggle_statechart()
    service.registry.register_statechart("toggle", toggle_chart)
    
    # Use managed machine context
    async with managed_machine(service, "toggle", {"user": "demo"}) as machine:
        # Send events
        await service.send_event(machine.id, SendEventRequest(event="TOGGLE"))
        
        # Check state
        updated_machine = await service.get_machine(machine.id)
        print(f"Machine state: {updated_machine.state.name}")
        print(f"Active states: {updated_machine.get_active_states()}")
    
    # Machine is automatically cleaned up when exiting the context