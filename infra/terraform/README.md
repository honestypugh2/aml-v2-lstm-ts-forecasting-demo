# Azure ML Secure Workspace – Managed VNet Isolation (Terraform)

End-to-end Terraform deployment of an Azure Machine Learning workspace secured
with **Workspace Managed Virtual Network** isolation, following the
[Tutorial: Create a secure workspace with a managed virtual network](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-create-secure-workspace?view=azureml-api-2)
and the [Managed VNet isolation guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-managed-network?view=azureml-api-2).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Azure Resource Group                                                       │
│                                                                             │
│  ┌──────────────────────────────────┐  ┌──────────────────────────────────┐ │
│  │  AML Workspace                   │  │  Jumpbox VNet (10.30.0.0/16)    │ │
│  │  managed_network:                │  │                                  │ │
│  │    AllowInternetOutbound         │  │  AzureBastionSubnet (10.30.2.0) │ │
│  │                                  │  │    └─ Bastion Host              │ │
│  │  Managed VNet (hidden, auto)     │  │                                  │ │
│  │  ┌────────────────────────────┐  │  │  snet-jumpbox (10.30.1.0)       │ │
│  │  │ Auto PEs to Storage/KV/ACR│  │  │    ├─ Jumpbox VM (Entra-joined) │ │
│  │  │ (for compute traffic)     │  │  │    ├─ Workspace PE ──────────┐  │ │
│  │  └────────────────────────────┘  │◄─┤    ├─ Storage blob PE       │  │ │
│  │                                  │PE│    ├─ Storage file PE       │  │ │
│  │  cpu-cluster (triggers managed   │  │    ├─ Key Vault PE          │  │ │
│  │  VNet provisioning)              │  │    └─ ACR PE                │  │ │
│  └──────────────────────────────────┘  └──────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │ Storage Account   │ │ Key Vault    │ │ ACR (Premium)│ │ App Insights │  │
│  │ default_action:   │ │ default:Deny │ │ public:false │ │ + Log Analyt │  │
│  │ Deny              │ │              │ │              │ │              │  │
│  └───────────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Two sets of private endpoints exist:**
1. **Managed VNet PEs** (auto-created by AML): for compute traffic to
   Storage/KV/ACR inside the hidden managed network.
2. **Jumpbox VNet PEs** (Terraform-managed): for jumpbox/Studio browser
   access to the workspace and its backing services.

## Resources created

| Resource | Purpose |
|---|---|
| Resource Group | Container for all resources |
| Log Analytics Workspace | Backend for Application Insights |
| Application Insights | AML workspace telemetry |
| Storage Account (`default_action=Deny`) | AML default datastore, firewall-restricted |
| Azure Container Registry (Premium, `public_network_access=false`) | Docker images for training environments |
| Key Vault (`default_action=Deny`) | Secrets/keys for AML, firewall-restricted |
| AML Workspace (`managed_network`) | Workspace with managed VNet isolation (`AllowInternetOutbound`) |
| AML Compute Cluster | `cpu-cluster` — triggers managed VNet provisioning |
| **Workspace identity RBAC** | `Storage Blob Data Contributor`, `Storage File Data Privileged Contributor`, `AcrPush`, `Key Vault Administrator` on backing services |
| **Deployer RBAC** | `Contributor` on workspace (includes `privateEndpointConnections/read` + `write` for managed VNet provisioning) |
| Jumpbox VNet + subnets | User-managed VNet for Bastion + workspace inbound access |
| Azure Bastion Host (Standard) | Secure browser-based RDP access to the jumpbox |
| Jumpbox Windows VM + Entra ID join | Windows desktop with `AADLoginForWindows` extension for Conditional Access compliance |
| `Virtual Machine Administrator Login` | Entra RBAC for jumpbox RDP |
| Workspace Private Endpoint | Inbound access to the workspace from the jumpbox VNet |
| Storage blob + file Private Endpoints | Jumpbox/Studio access to blob and file share data |
| Key Vault Private Endpoint | Jumpbox/Studio access to Key Vault |
| ACR Private Endpoint | Jumpbox/Studio access to Container Registry |
| Private DNS Zones | `privatelink.api.azureml.ms`, `privatelink.notebooks.azure.net`, `privatelink.blob.core.windows.net`, `privatelink.file.core.windows.net`, `privatelink.vaultcore.azure.net`, `privatelink.azurecr.io` |

## Required RBAC permissions

The Azure identity used to deploy this infrastructure needs the following
permissions. The `Contributor` role on the resource group satisfies all of
these:

| Action | Why |
|---|---|
| `Microsoft.MachineLearningServices/workspaces/privateEndpointConnections/read` | Read managed VNet private endpoint connections |
| `Microsoft.MachineLearningServices/workspaces/privateEndpointConnections/write` | Create/approve managed VNet private endpoint connections |
| Standard `Contributor` actions | Create all resources (VMs, PEs, DNS, etc.) |
| `Microsoft.Authorization/roleAssignments/write` | Assign RBAC roles (requires `Owner` or `User Access Administrator`) |

> **Note:** If your identity only has `Contributor`, you will also need
> `User Access Administrator` on the resource group to create the role
> assignments for the workspace identity and jumpbox.

## How Managed VNet isolation works

Azure ML creates and manages a hidden VNet for all managed compute (clusters,
instances, online endpoints). The workspace's `managed_network.isolation_mode`
controls outbound traffic from that managed VNet:

| Mode | Behavior |
|---|---|
| `AllowInternetOutbound` (default) | All internet outbound allowed; private endpoints auto-created to workspace dependencies with public access disabled |
| `AllowOnlyApprovedOutbound` | Only traffic matching approved outbound rules (service tags, FQDNs, private endpoints) is allowed |

The managed VNet and its private endpoints are **provisioned on first compute
creation**, which is why this package includes a compute cluster. For more
details, see [Manually provision a managed VNet](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-managed-network?view=azureml-api-2#manually-provision-a-managed-vnet).

## Prerequisites

* [Terraform >= 1.5](https://developer.hashicorp.com/terraform/install)
* Azure CLI authenticated: `az login`
* Subscription selected: `az account set --subscription <SUB_ID>`
* Providers: `azurerm ~> 4.0`, `azapi ~> 2.0`
* A strong admin password (12+ chars, mixed case, digit, special) for the Windows jumpbox
* `Owner` or `Contributor` + `User Access Administrator` on the target resource group/subscription

## Deploy

```bash
cd infra/terraform

# 1. Copy and edit variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your subscription ID and globally unique names

# 2. Deploy
terraform init
terraform plan -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars
```

## Access the workspace

Because `public_network_access = Disabled`, the workspace can only be reached
through the **private endpoint** on the jumpbox VNet. The jumpbox is a
**Windows Server 2022** VM with a desktop GUI (Desktop Experience), so you can
use a browser, VS Code, and CLI tools directly on the VM.

The jumpbox is **Entra ID-joined** via the `AADLoginForWindows` VM extension.
This is required for corporate Conditional Access policies that restrict
browser access (e.g., Azure ML Studio) to compliant/managed devices.

| Approach | Use case |
|---|---|
| [Option 1](#option-1--rdp-via-azure-bastion-browser-based) | Full desktop: ML Studio, VS Code, CLI — no local port setup |
| [Option 2](#option-2--rdp-via-bastion-tunnel--local-rdp-client) | Same desktop, but using your preferred local RDP client |
| [Option 3](#option-3--cli-only-via-bastion-tunnel) | Headless CLI/SDK access without a desktop |

### Common prerequisites

* `terraform apply` completed successfully (Bastion host, jumpbox VM, and
  private endpoint are all deployed).
* Azure CLI >= 2.32 with the `bastion` extension:
  ```bash
  az extension add --name bastion
  ```

### Starting the jumpbox VM

The jumpbox may be deallocated (stopped) to save costs. Start it before
connecting:

```bash
az vm start \
  --name <aml_workspace_name>-jumpbox \
  --resource-group <resource_group_name>

# Verify it's running
az vm show \
  --name <aml_workspace_name>-jumpbox \
  --resource-group <resource_group_name> \
  --show-details --query "powerState" -o tsv
# Expected: VM running
```

---

### Option 1 — RDP via Azure Bastion (browser-based)

The simplest approach — no local tooling beyond Azure Portal.

1. Go to the [Azure Portal](https://portal.azure.com).
2. Navigate to the jumpbox VM (`<aml_workspace_name>-jumpbox`).
3. Click **Connect** → **Connect via Bastion**.
4. Enter:
   * **Username:** `azureuser`
   * **Password:** *(the `admin_password` from your `terraform.tfvars`)*
5. Click **Connect**. A browser-based RDP session opens.

#### Accessing Azure ML Studio (Conditional Access)

The jumpbox is Entra ID-joined at the device level. To get a Primary Refresh
Token (PRT) so Conditional Access passes in the browser:

1. On the jumpbox open **Settings** → **Accounts** → **Access work or school**.
2. If your Entra account is not listed, click **Connect** and sign in with
   your `@microsoft.com` (or corporate) account.
3. Open **Microsoft Edge** → sign in to Edge with the same corporate account.
4. Enable **"Allow my organization to manage my device"** if prompted.
5. Navigate to `https://ml.azure.com`.

#### Verify DNS resolution

From PowerShell on the jumpbox, confirm private DNS is working:

```powershell
nslookup <workspace-name>.api.azureml.ms
nslookup <storage-account>.blob.core.windows.net
nslookup <storage-account>.file.core.windows.net
nslookup <keyvault-name>.vault.azure.net
nslookup <acr-name>.azurecr.io
# All should resolve to 10.30.1.x private IPs
```

---

### Option 2 — RDP via Bastion tunnel + local RDP client

For a better RDP experience (clipboard, multiple monitors, etc.), tunnel the
RDP port through Bastion.

#### Open the Bastion tunnel

```bash
az network bastion tunnel \
  --name <BASTION_NAME> \
  --resource-group <RESOURCE_GROUP> \
  --target-resource-id <JUMPBOX_VM_RESOURCE_ID> \
  --resource-port 3389 \
  --port 3390
```

Get the VM resource ID from Terraform output:

```bash
cd infra/terraform
terraform output -raw jumpbox_vm_id
```

#### Connect with your RDP client

* **Windows:** Run `mstsc` → enter `127.0.0.1:3390`
* **macOS:** Use [Microsoft Remote Desktop](https://apps.apple.com/app/microsoft-remote-desktop/id1295203466)
* **Linux:** `xfreerdp /v:127.0.0.1:3390 /u:azureuser /size:1920x1080`

Enter the `admin_password` when prompted.

---

### Option 3 — CLI only via Bastion tunnel

For headless automation without a desktop.

#### Open the Bastion tunnel

```bash
az network bastion tunnel \
  --name <BASTION_NAME> \
  --resource-group <RESOURCE_GROUP> \
  --target-resource-id <JUMPBOX_VM_RESOURCE_ID> \
  --resource-port 3389 \
  --port 3390
```

#### Connect via PowerShell Remoting over RDP

You can use an RDP client to launch just PowerShell, or use `az network bastion
rdp` (Windows only) for a quick session:

```bash
az network bastion rdp \
  --name <BASTION_NAME> \
  --resource-group <RESOURCE_GROUP> \
  --target-resource-id <JUMPBOX_VM_RESOURCE_ID>
```

> **Note:** `az network bastion rdp` requires a Windows client with WinDLL.
> On Linux/macOS, use the tunnel approach above instead.

---

## Setting up the jumpbox (first time)

After connecting to the Windows jumpbox via RDP, install the development tools.
Open **PowerShell** (search "PowerShell" in Start) and run the steps below.

### 1. Install VS Code

VS Code runs natively on the Windows jumpbox and supports all extensions
(Python, Jupyter, Azure ML, etc.):

```powershell
# Download and install VS Code (system installer, silent)
Invoke-WebRequest -Uri "https://update.code.visualstudio.com/latest/win32-x64/stable" -OutFile "$env:TEMP\vscode-setup.exe"
Start-Process -FilePath "$env:TEMP\vscode-setup.exe" -ArgumentList "/verysilent /mergetasks=!runcode,addcontextmenufiles,addcontextmenufolders,addtopath" -Wait
Remove-Item "$env:TEMP\vscode-setup.exe"

# Refresh PATH (or restart terminal)
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

# Install recommended extensions
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
code --install-extension ms-toolsai.vscode-ai
```

> **Note:** VS Code is fully supported on Windows Server 2022. All extensions
> (Python, Jupyter, Azure ML, GitHub Copilot, etc.) work as on any Windows
> desktop.

### 2. Install Python

```powershell
# Download and install Python 3.11
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile "$env:TEMP\python-installer.exe"
Start-Process -FilePath "$env:TEMP\python-installer.exe" -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1" -Wait
Remove-Item "$env:TEMP\python-installer.exe"

# Refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

# Verify
python --version
# Expected: Python 3.11.9
```

### 3. Install uv (Python package manager)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

# Verify
uv --version
```

### 4. Install Git

```powershell
# Download and install Git
Invoke-WebRequest -Uri "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.2/Git-2.47.1.2-64-bit.exe" -OutFile "$env:TEMP\git-installer.exe"
Start-Process -FilePath "$env:TEMP\git-installer.exe" -ArgumentList "/VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -Wait
Remove-Item "$env:TEMP\git-installer.exe"

# Refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

git --version
```

### 5. Install Azure CLI

```powershell
# Download and install Azure CLI
Invoke-WebRequest -Uri "https://aka.ms/installazurecliwindowsx64" -OutFile "$env:TEMP\az-cli.msi"
Start-Process msiexec.exe -ArgumentList "/I $env:TEMP\az-cli.msi /quiet" -Wait
Remove-Item "$env:TEMP\az-cli.msi"

# Refresh PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

az --version
az extension add --name ml --yes
```

### 6. Clone the project and install dependencies

```powershell
cd $env:USERPROFILE

git clone https://github.com/honestypugh2/aml-v2-lstm-ts-forecasting-demo.git
cd aml-v2-lstm-ts-forecasting-demo

# Install all Python dependencies (creates .venv automatically)
uv sync

# Verify key packages
uv run python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### 7. Authenticate with Azure

```powershell
az login

# Set your subscription
az account set --subscription "<SUB_ID>"

# Verify workspace access
az ml workspace show --name <aml_workspace_name> --resource-group <resource_group_name>
```

---

## Using Azure ML Studio

Once set up, open **Microsoft Edge** (pre-installed on Windows Server 2022)
and navigate to:

```
https://ml.azure.com
```

Sign in with your Microsoft Entra ID credentials. MFA works normally in Edge.
Select your workspace or navigate directly:

```
https://ml.azure.com/home?tid=<TENANT_ID>&wsid=<WORKSPACE_RESOURCE_ID>
```

DNS resolution for `*.api.azureml.ms` and `*.notebooks.azure.net` resolves to
private IPs automatically because the private DNS zones are linked to the
jumpbox VNet.

## Using VS Code on the jumpbox

After installing VS Code (Step 1 above), launch it from the Start menu or
run `code .` in the project directory. You can:

* Open the project folder and use the integrated terminal
* Run Jupyter notebooks directly (with the Jupyter extension)
* Use the Azure ML extension to manage experiments, compute, and models
* Use GitHub Copilot and all other extensions

---

## Development workflow (on the jumpbox)

```powershell
cd $env:USERPROFILE\aml-v2-lstm-ts-forecasting-demo

# Run training jobs
uv run python src/azure_ml_training/submit_training_job.py `
  --compute-name cpu-cluster `
  --experiment-name lstm-forecasting

# Run local training
uv run python src/training/train_lstm.py

# Run tests
uv run pytest tests/ -v

# Use the Azure ML SDK
uv run python -c @"
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id='<SUB_ID>',
    resource_group_name='<RG>',
    workspace_name='<WORKSPACE>',
)
for c in ml_client.compute.list():
    print(c.name, c.type)
"@
```

## Relationship to `infra/azure-ml-vnet`

| Directory | Purpose |
|---|---|
| `infra/terraform/` | **Full secure workspace** — AML workspace + all dependencies + managed VNet + jumpbox/Bastion |
| `infra/azure-ml-vnet/` | **BYO VNet foundation** — standalone VNet, subnet, NSG rules, and managed identity for custom/existing network topologies |

## References

* [Tutorial: Create a secure workspace with a managed virtual network](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-create-secure-workspace?view=azureml-api-2)
* [Workspace Managed Virtual Network Isolation](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-managed-network?view=azureml-api-2)
* [azureml-examples: workspace-managed-network.ipynb](https://github.com/Azure/azureml-examples/blob/main/sdk/python/resources/workspace/workspace-managed-network.ipynb)