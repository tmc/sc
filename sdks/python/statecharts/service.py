"""
gRPC service client for statechart operations.

This module provides a client interface for interacting with statechart services
over gRPC, including machine creation and execution.
"""

from typing import Optional, Dict, Any
import grpc

from .core import Statechart, Machine, MachineState
from .generated import (
    StatechartServiceStub,
    CreateMachineRequest,
    CreateMachineResponse,
    StepRequest,
    StepResponse,
    StatechartRegistry,
)


class StatechartService:
    """
    Client for interacting with a statechart service over gRPC.
    
    This class provides a high-level interface for creating machines
    and stepping through statechart execution.
    """
    
    def __init__(self, channel: grpc.Channel):
        """
        Initialize the service client.
        
        Args:
            channel: gRPC channel to the statechart service
        """
        self.stub = StatechartServiceStub(channel)
    
    def create_machine(
        self, 
        statechart_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Machine:
        """
        Create a new machine instance from a statechart.
        
        Args:
            statechart_id: ID of the statechart to create a machine from
            context: Optional initial context for the machine
            
        Returns:
            The created Machine instance
            
        Raises:
            grpc.RpcError: If the gRPC call fails
        """
        from google.protobuf.struct_pb2 import Struct
        
        pb_context = None
        if context:
            pb_context = Struct()
            pb_context.update(context)
        
        request = CreateMachineRequest(
            statechart_id=statechart_id,
            context=pb_context
        )
        
        response: CreateMachineResponse = self.stub.CreateMachine(request)
        return Machine.from_pb(response.machine)
    
    def step(
        self, 
        statechart_id: str, 
        event: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[Machine, Any]:
        """
        Step a statechart through one iteration with an event.
        
        Args:
            statechart_id: ID of the statechart to step
            event: Event to process
            context: Optional context data for the event
            
        Returns:
            Tuple of (updated_machine, step_result)
            
        Raises:
            grpc.RpcError: If the gRPC call fails
        """
        from google.protobuf.struct_pb2 import Struct
        
        pb_context = None
        if context:
            pb_context = Struct()
            pb_context.update(context)
        
        request = StepRequest(
            statechart_id=statechart_id,
            event=event,
            context=pb_context
        )
        
        response: StepResponse = self.stub.Step(request)
        machine = Machine.from_pb(response.machine)
        
        return machine, response.result


class StatechartClient:
    """
    High-level client for statechart operations.
    
    This class provides a convenient interface for common statechart operations
    with connection management and error handling.
    """
    
    def __init__(self, server_address: str = "localhost:50051"):
        """
        Initialize the client.
        
        Args:
            server_address: Address of the statechart service
        """
        self.server_address = server_address
        self.channel: Optional[grpc.Channel] = None
        self.service: Optional[StatechartService] = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def connect(self):
        """Establish connection to the service."""
        self.channel = grpc.insecure_channel(self.server_address)
        self.service = StatechartService(self.channel)
    
    def close(self):
        """Close the connection."""
        if self.channel:
            self.channel.close()
            self.channel = None
            self.service = None
    
    def create_machine(
        self, 
        statechart_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Machine:
        """
        Create a new machine instance.
        
        Args:
            statechart_id: ID of the statechart
            context: Optional initial context
            
        Returns:
            The created Machine instance
        """
        if not self.service:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        return self.service.create_machine(statechart_id, context)
    
    def step(
        self, 
        statechart_id: str, 
        event: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[Machine, Any]:
        """
        Step a statechart with an event.
        
        Args:
            statechart_id: ID of the statechart
            event: Event to process
            context: Optional event context
            
        Returns:
            Tuple of (updated_machine, step_result)
        """
        if not self.service:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        return self.service.step(statechart_id, event, context)


# Convenience functions for common operations

def create_insecure_client(server_address: str = "localhost:50051") -> StatechartClient:
    """
    Create an insecure (non-TLS) statechart client.
    
    Args:
        server_address: Address of the statechart service
        
    Returns:
        A StatechartClient instance
        
    Example:
        >>> with create_insecure_client() as client:
        ...     machine = client.create_machine("my-statechart")
    """
    return StatechartClient(server_address)


def create_secure_client(
    server_address: str, 
    credentials: grpc.ChannelCredentials
) -> StatechartClient:
    """
    Create a secure (TLS) statechart client.
    
    Args:
        server_address: Address of the statechart service
        credentials: gRPC channel credentials
        
    Returns:
        A StatechartClient instance with secure connection
        
    Example:
        >>> import grpc
        >>> creds = grpc.ssl_channel_credentials()
        >>> with create_secure_client("secure.example.com:443", creds) as client:
        ...     machine = client.create_machine("my-statechart")
    """
    client = StatechartClient.__new__(StatechartClient)
    client.server_address = server_address
    client.channel = None
    client.service = None
    
    def connect():
        client.channel = grpc.secure_channel(server_address, credentials)
        client.service = StatechartService(client.channel)
    
    client.connect = connect
    return client


class AsyncStatechartService:
    """
    Async client for interacting with a statechart service over gRPC.
    
    This class provides an async interface for statechart operations.
    """
    
    def __init__(self, channel):
        """
        Initialize the async service client.
        
        Args:
            channel: Async gRPC channel to the statechart service
        """
        from .generated import StatechartServiceStub
        self.stub = StatechartServiceStub(channel)
    
    async def create_machine(
        self, 
        statechart_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Machine:
        """
        Async create a new machine instance from a statechart.
        
        Args:
            statechart_id: ID of the statechart to create a machine from
            context: Optional initial context for the machine
            
        Returns:
            The created Machine instance
        """
        from google.protobuf.struct_pb2 import Struct
        
        pb_context = None
        if context:
            pb_context = Struct()
            pb_context.update(context)
        
        request = CreateMachineRequest(
            statechart_id=statechart_id,
            context=pb_context
        )
        
        response: CreateMachineResponse = await self.stub.CreateMachine(request)
        return Machine.from_pb(response.machine)
    
    async def step(
        self, 
        statechart_id: str, 
        event: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[Machine, Any]:
        """
        Async step a statechart through one iteration with an event.
        
        Args:
            statechart_id: ID of the statechart to step
            event: Event to process
            context: Optional context data for the event
            
        Returns:
            Tuple of (updated_machine, step_result)
        """
        from google.protobuf.struct_pb2 import Struct
        
        pb_context = None
        if context:
            pb_context = Struct()
            pb_context.update(context)
        
        request = StepRequest(
            statechart_id=statechart_id,
            event=event,
            context=pb_context
        )
        
        response: StepResponse = await self.stub.Step(request)
        machine = Machine.from_pb(response.machine)
        
        return machine, response.result


class AsyncStatechartClient:
    """
    High-level async client for statechart operations.
    """
    
    def __init__(self, server_address: str = "localhost:50051"):
        """
        Initialize the async client.
        
        Args:
            server_address: Address of the statechart service
        """
        self.server_address = server_address
        self.channel = None
        self.service: Optional[AsyncStatechartService] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def connect(self):
        """Establish async connection to the service."""
        import grpc.aio
        self.channel = grpc.aio.insecure_channel(self.server_address)
        self.service = AsyncStatechartService(self.channel)
    
    async def close(self):
        """Close the async connection."""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.service = None
    
    async def create_machine(
        self, 
        statechart_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Machine:
        """
        Async create a new machine instance.
        
        Args:
            statechart_id: ID of the statechart
            context: Optional initial context
            
        Returns:
            The created Machine instance
        """
        if not self.service:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        return await self.service.create_machine(statechart_id, context)
    
    async def step(
        self, 
        statechart_id: str, 
        event: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[Machine, Any]:
        """
        Async step a statechart with an event.
        
        Args:
            statechart_id: ID of the statechart
            event: Event to process
            context: Optional event context
            
        Returns:
            Tuple of (updated_machine, step_result)
        """
        if not self.service:
            raise RuntimeError("Client not connected. Call connect() first.")
        
        return await self.service.step(statechart_id, event, context)


# Async convenience functions

async def create_async_insecure_client(server_address: str = "localhost:50051") -> AsyncStatechartClient:
    """
    Create an async insecure (non-TLS) statechart client.
    
    Args:
        server_address: Address of the statechart service
        
    Returns:
        An AsyncStatechartClient instance
        
    Example:
        >>> async with create_async_insecure_client() as client:
        ...     machine = await client.create_machine("my-statechart")
    """
    return AsyncStatechartClient(server_address)