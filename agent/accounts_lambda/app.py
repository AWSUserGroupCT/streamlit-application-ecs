import json
import boto3

def lambda_handler(event, context):
    agent = event['agent']
    actionGroup = event['actionGroup']
    function = event['function']
    parameters = event.get('parameters', [])
    
    def get_account_number(account_name):
        # query dynamo db for account number using account name
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('AWSOrgAccounts')
        response = table.scan(
            FilterExpression=boto3.dynamodb.conditions.Attr('AccountName').eq(account_name)
        )
        return response['Items'][0]['Id']
        
    def get_account_name(account_number):
        # query dynamo db for account name using account number
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('AWSOrgAccounts')
        response = table.get_item(
            Key={
                'Id': account_number
            }
        )
        return response['Item']['AccountName']

    
    params_dict = {param['name'].lower(): param['value'] for param in parameters}
    print("Params: {}".format(params_dict))

    responseBody = {}  # Initialize responseBody with a default value

    if function == 'get_account_number':
        account_name = params_dict['account_name']
        account_number = get_account_number(account_name)
        responseBody = {
            "TEXT": {
                "body": "The account number for {} is {}".format(account_name, account_number)
            }
        }
    elif function == 'get_account_name':
        account_number = params_dict['account_number']
        account_name = get_account_name(account_number)
        responseBody = {
            "TEXT": {
                "body": "The account name for {} is {}".format(account_number, account_name)
            }
        }

    action_response = {
        'actionGroup': actionGroup,
        'function': function,
        'functionResponse': {
            'responseBody': responseBody
        }
    }

    function_response = {'response': action_response, 'messageVersion': event['messageVersion']}
    print("Response: {}".format(function_response))

    return function_response
