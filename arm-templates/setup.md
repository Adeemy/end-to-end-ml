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
8. Deploy the ARM template to create the needed resources in the Azure resource group
   - `az deployment group create --name <YourResourcesDeployment> --resource-group <ResourceGroupName> --template-file ./arm-templates/azure_train_resource.json`
9. Create Service Principal to access Azure resources remotely
   - `az ad sp create-for-rbac --name "sp-e2e" --role contributor --scopes /subscriptions/<YourSubscriptionID>/resourceGroups/<ResourceGroupName>`
10. Copy the output that includes the credentials to GitHub Actions Secrets (don't store anywhere else other than Azure Key Vault)
11. Store the Service Principal credentials (clientId and clientSecret) to Azure Key Vault to be accessed by Python SDK
    - `az keyvault set-policy --name kv-e2e-ml-jpvfucwdep4dc --spn a1fb099b-6d33-4aba-a9a0-24e5d491b17d --secret-permissions set`
    - `az login --service-principal -u a1fb099b-6d33-4aba-a9a0-24e5d491b17d -p eaH8Q~edxvpDsDFjDJ.16.hyPVtISP-Wei2iAbgI --tenant bd7d8290-77d5-4539-b7f6-59802055a290`
    - `az keyvault secret set --vault-name kv-e2e-ml-jpvfucwdep4dc --name ServicePrincipalID --value a1fb099b-6d33-4aba-a9a0-24e5d491b17d`
    - `az keyvault secret set --vault-name kv-e2e-ml-jpvfucwdep4dc --name ServicePrincipalPassword --value eaH8Q~edxvpDsDFjDJ.16.hyPVtISP-Wei2iAbgI`
