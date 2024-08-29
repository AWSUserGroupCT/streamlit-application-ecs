import streamlit as st
import logging
from botocore.exceptions import ClientError
import boto3

"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with the Agents for Amazon
Bedrock Runtime client to send prompts to an agent to process and respond to.
"""

logger = logging.getLogger(__name__)

# snippet-start:[python.example_code.bedrock-agent-runtime.BedrockAgentsRuntimeWrapper.class]
# snippet-start:[python.example_code.bedrock-agent-runtime.BedrockAgentRuntimeWrapper.decl]
class BedrockAgentRuntimeWrapper:
    """Encapsulates Agents for Amazon Bedrock Runtime actions."""

    def __init__(self, runtime_client):
        """
        :param runtime_client: A low-level client representing the Agents for Amazon
                               Bedrock Runtime. Describes the API operations for running
                               inferences using Bedrock Agents.
        """
        self.agents_runtime_client = runtime_client

    def invoke_agent(self, agent_id, agent_alias_id, session_id, prompt):
        """
        Sends a prompt for the agent to process and respond to.

        :param agent_id: The unique identifier of the agent to use.
        :param agent_alias_id: The alias of the agent to use.
        :param session_id: The unique identifier of the session. Use the same value across requests
                           to continue the same conversation.
        :param prompt: The prompt that you want Claude to complete.
        :return: Inference response from the model.
        """

        try:
            # Note: The execution time depends on the foundation model, complexity of the agent,
            # and the length of the prompt. In some cases, it can take up to a minute or more to
            # generate a response.
            response = self.agents_runtime_client.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                inputText=prompt,
            )

            completion = ""

            for event in response.get("completion"):
                chunk = event["chunk"]
                completion = completion + chunk["bytes"].decode()

        except ClientError as e:
            logger.error(f"Couldn't invoke agent. {e}")
            raise

        return completion
    # snippet-end:[python.example_code.bedrock-agent-runtime.BedrockAgentRuntimeWrapper.decl]

    # snippet-start:[python.example_code.bedrock-agent-runtime.InvokeAgent]
    def invoke_agent(self, agent_id, agent_alias_id, session_id, prompt):
        """
        Sends a prompt for the agent to process and respond to.

        :param agent_id: The unique identifier of the agent to use.
        :param agent_alias_id: The alias of the agent to use.
        :param session_id: The unique identifier of the session. Use the same value across requests
                           to continue the same conversation.
        :param prompt: The prompt that you want Claude to complete.
        :return: Inference response from the model.
        """

        try:
            # Note: The execution time depends on the foundation model, complexity of the agent,
            # and the length of the prompt. In some cases, it can take up to a minute or more to
            # generate a response.
            response = self.agents_runtime_client.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                inputText=prompt,
            )

            completion = ""

            for event in response.get("completion"):
                chunk = event["chunk"]
                completion = completion + chunk["bytes"].decode()

        except ClientError as e:
            logger.error(f"Couldn't invoke agent. {e}")
            raise

        return completion

st.title("Chat with an Agent")
question = st.text_input("Sign up for a talk at the Conneticut AWS UserGroup")

if question:
    try:
        # snippet-start:[python.example_code.bedrock-agent-runtime.CreateClient]

        # Create a Boto3 client for the Agents for Amazon Bedrock Runtime
        agents_runtime_client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
        # snippet-end:[python.example_code.bedrock-agent-runtime.CreateClient]

        # snippet-start:[python.example_code.bedrock-agent-runtime.CreateWrapper]
        bedrock_agent = BedrockAgentRuntimeWrapper(agents_runtime_client)
        # snippet-end:[python.example_code.bedrock-agent-runtime.CreateWrapper]

        # snippet-start:[python.example_code.bedrock-agent-runtime.InvokeAgent]
        agent_id = "ESE2FSH4QM"
        agent_alias_id = "0RPN12ID7K"
        session_id = "example-session-id"

        response = bedrock_agent.invoke_agent(agent_id, agent_alias_id, session_id, question)
        # snippet-end:[python.example_code.bedrock-agent-runtime.InvokeAgent]

        st.write(response)

    except ClientError as e:
        st.error(f"Couldn't process the question: {e}")