Steps to deploy ARM template to create Azure resources for training:

1. Install Azure Tools VS code extension (extension ID ms-vscode.vscode-node-azure-pack)
2. Install Azure CLI and ML workspace extension
   - `curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash`
   - `az extension add -n ml -y`
3. Log in to Azure subscription
   - `az login`
4. Set the active subscription
   - `az account set --subscription <Subscription-ID>`
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
   - `az deployment group create --name <ResourcesDeployment> --resource-group <ResourceGroupName> --template-file ./infrastructure/azure_train_resource.json`
9. Create Service Principal to access Azure resources remotely
   - `az ad sp create-for-rbac --name "sp-e2e" --role contributor --scopes /subscriptions/<Subscription-ID>/resourceGroups/<ResourceGroupName>`
10. Copy the output that includes the credentials to GitHub Actions Secrets (don't store anywhere else other than Azure Key Vault)
11. Store the Service Principal credentials (clientId and clientSecret) to Azure Key Vault to be accessed by Python SDK
    - `az keyvault set-policy --name <KeyVaultName> --spn <ServicePrincipalID> --secret-permissions set`
    - `az login --service-principal -u <ServicePrincipalID> -p <ServicePrincipalPassword> --tenant <TenantID>`
    - `az keyvault secret set --vault-name <KeyVaultName> --name ServicePrincipalID --value <ServicePrincipalID>`
    - `az keyvault secret set --vault-name <KeyVaultName> --name ServicePrincipalPassword --value <ServicePrincipalPassword>`
12. Store Azure credentials for login in the following format:
    {
    "clientId": <APP_ID>,
    "clientSecret": <SP_PWD>,
    "subscriptionId": <Subscription-ID>,
    "tenantId": <TENANT_ID>
    }
13. (Optional) Grant permissions to access secrets in key vault to specific user
    - `az keyvault set-policy --name <KeyVaultName> --upn <UserPrincipalName> --secret-permissions get list --key-permissions get list --certificate-permissions get list`
