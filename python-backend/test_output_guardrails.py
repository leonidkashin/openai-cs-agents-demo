#!/usr/bin/env python3
"""
Test script to verify that output guardrails are now included in the API response.
"""

import asyncio
import json
import sys
import os

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from api import ChatRequest, chat_endpoint

async def test_output_guardrails():
    """Test that output guardrails are included in the API response."""
    
    # Create a test request that should trigger output guardrails
    # This should go to the seat booking agent which has tov_guardrail as output guardrail
    test_request = ChatRequest(
        conversation_id="",  # New conversation
        message="привет, хочу поменять кресло"  # Russian message about changing seat
    )
    
    try:
        print("Testing output guardrails inclusion...")
        
        # Call the chat endpoint
        response = await chat_endpoint(test_request)
        
        # Convert to dict for easier inspection
        response_dict = response.model_dump()
        
        # Check if output_guardrails field exists
        if 'output_guardrails' in response_dict:
            print("✓ output_guardrails field is present in the response")
            print(f"Number of output guardrails: {len(response_dict['output_guardrails'])}")
            
            if response_dict['output_guardrails']:
                print("✓ Output guardrails found!")
                for i, og in enumerate(response_dict['output_guardrails']):
                    print(f"  Output Guardrail {i+1}:")
                    print(f"    Name: {og.get('name', 'N/A')}")
                    print(f"    Reasoning: {og.get('reasoning', 'N/A')[:100]}...")
                    print(f"    Final Text: {og.get('final_text', 'N/A')[:100]}...")
                    print(f"    Tripwire Triggered: {og.get('tripwire_triggered', 'N/A')}")
            else:
                print("⚠ output_guardrails field is empty")
        else:
            print("✗ output_guardrails field is missing from the response")
            
        # Check agents metadata for output_guardrails
        print("\nChecking agents metadata:")
        for agent in response_dict.get('agents', []):
            output_guardrails = agent.get('output_guardrails', [])
            if output_guardrails:
                print(f"✓ Agent '{agent['name']}' has output guardrails: {output_guardrails}")
        
        # Print the full response for inspection (truncated)
        print("\n" + "="*50)
        print("Sample response structure:")
        print(json.dumps({
            'conversation_id': response_dict.get('conversation_id', 'N/A'),
            'current_agent': response_dict.get('current_agent', 'N/A'),
            'messages_count': len(response_dict.get('messages', [])),
            'events_count': len(response_dict.get('events', [])),
            'guardrails_count': len(response_dict.get('guardrails', [])),
            'output_guardrails_count': len(response_dict.get('output_guardrails', [])),
        }, indent=2))
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing output guardrails: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_output_guardrails())
    if success:
        print("\n✓ Test completed successfully")
    else:
        print("\n✗ Test failed")
        sys.exit(1)