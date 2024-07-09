Steps to deploy ARM template to create Azure resources for training:

1. Install Azure Tools VS code extension (extension ID ms-vscode.vscode-node-azure-pack)
2. Install Azure CLI
   - `curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash`
3. Log in to Azure subscription
   - `az login`
4. Set the active subscription
   - `az account set --subscription "YourSubscriptionID"`
5. Create Azure resource group
   - `az group create --name <ResourceGroupName> --location <ResourceGroupLocation>`
6. Create ARM template using Quick Start as illustrated in Azure [Docs](https://learn.microsoft.com/en-us/azure/azure-resource-manager/templates/quickstart-create-templates-use-the-portal)
7. Add the following resource definition for the compute cluster used for training:
   ```json
   {
     "type": "Microsoft.MachineLearningServices/workspaces/computes",
     "apiVersion": "2020-08-01",
     "name": "[concat(parameters('workspaceName'), '/<computeClusterName>')]",
     "location": "[parameters('location')]",
     "dependsOn": [
       "[resourceId('Microsoft.MachineLearningServices/workspaces', parameters('workspaceName'))]"
     ],
     "properties": {
       "computeType": "AmlCompute",
       "properties": {
         "vmSize": "STANDARD_DS3_V2",
         "scaleSettings": {
           "minNodeCount": 0,
           "maxNodeCount": 4,
           "idleSecondsBeforeScaledown": "120",
           "enableAutoScale": true
         }
       }
     }
   }
   ```

