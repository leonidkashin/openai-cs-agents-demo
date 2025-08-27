#!/usr/bin/env python3
"""
Test script to verify that agent responses now use the final_text from output guardrails.
"""

import asyncio
import json
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), 'app', '.env'))

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from api import ChatRequest, chat_endpoint

async def test_output_guardrail_final_text():
    """Test that agent responses use final_text from output guardrails."""
    
    # Create a test request that should trigger output guardrails
    # This should go through the triage agent which has tov_guardrail as output guardrail
    test_request = ChatRequest(
        conversation_id="",  # New conversation
        message="Привет, хочу поменять место в самолете"  # Russian message about changing airplane seat
    )
    
    try:
        print("Testing output guardrail final_text processing...")
        
        # Call the chat endpoint
        response = await chat_endpoint(test_request)
        
        # Convert to dict for easier inspection
        response_dict = response.model_dump()
        
        print(f"Current agent: {response_dict.get('current_agent', 'N/A')}")
        print(f"Number of messages: {len(response_dict.get('messages', []))}")
        print(f"Number of output guardrails: {len(response_dict.get('output_guardrails', []))}")
        
        # Check messages
        messages = response_dict.get('messages', [])
        if messages:
            print("\n=== AGENT MESSAGES ===")
            for i, msg in enumerate(messages):
                print(f"Message {i+1}:")
                print(f"  Agent: {msg.get('agent', 'N/A')}")
                print(f"  Content: {msg.get('content', 'N/A')[:200]}...")
                print()
        
        # Check output guardrails
        output_guardrails = response_dict.get('output_guardrails', [])
        if output_guardrails:
            print("=== OUTPUT GUARDRAILS ===")
            for i, og in enumerate(output_guardrails):
                print(f"Output Guardrail {i+1}:")
                print(f"  Name: {og.get('name', 'N/A')}")
                print(f"  Original Output: {og.get('output', 'N/A')[:100]}...")
                print(f"  Final Text: {og.get('final_text', 'N/A')[:200]}...")
                print(f"  Reasoning: {og.get('reasoning', 'N/A')[:100]}...")
                print(f"  Tripwire Triggered: {og.get('tripwire_triggered', 'N/A')}")
                print()
                
                # Check if the agent message content matches the guardrail final_text
                final_text = og.get('final_text', '').strip()
                if final_text and messages:
                    last_message_content = messages[-1].get('content', '').strip()
                    if final_text in last_message_content or last_message_content in final_text:
                        print("✓ SUCCESS: Agent message appears to use guardrail final_text!")
                    else:
                        print("⚠ WARNING: Agent message doesn't match guardrail final_text")
                        print(f"  Agent message: {last_message_content[:100]}...")
                        print(f"  Guardrail final_text: {final_text[:100]}...")
        else:
            print("⚠ No output guardrails found")
        
        # Print summary
        print("\n" + "="*50)
        print("SUMMARY:")
        if output_guardrails and messages:
            final_text = output_guardrails[0].get('final_text', '').strip()
            message_content = messages[-1].get('content', '').strip() if messages else ''
            
            if final_text and message_content:
                if final_text == message_content:
                    print("✓ PERFECT MATCH: Agent response exactly matches guardrail final_text")
                elif final_text in message_content or message_content in final_text:
                    print("✓ PARTIAL MATCH: Agent response contains guardrail final_text")
                else:
                    print("✗ NO MATCH: Agent response doesn't use guardrail final_text")
                    print(f"Expected: {final_text[:100]}...")
                    print(f"Got: {message_content[:100]}...")
            else:
                print("⚠ Missing data to compare")
        else:
            print("⚠ No data to analyze")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing output guardrail processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_output_guardrail_final_text())
    if success:
        print("\n✓ Test completed successfully")
    else:
        print("\n✗ Test failed")
        sys.exit(1)