data "azurerm_client_config" "current" {}

# --------------------------------------------------
# Resource Group
# --------------------------------------------------
resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

# --------------------------------------------------
# Jumpbox Virtual Network (workspace inbound access)
# --------------------------------------------------
resource "azurerm_virtual_network" "vnet" {
  name                = "${var.aml_workspace_name}-vnet"
  address_space       = ["10.30.0.0/16"]
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  tags                = var.tags
}

resource "azurerm_subnet" "snet_jumpbox" {
  name                 = "snet-jumpbox"
  resource_group_name  = azurerm_resource_group.rg.name
  virtual_network_name = azurerm_virtual_network.vnet.name
  address_prefixes     = ["10.30.1.0/24"]
}

# --------------------------------------------------
# Log Analytics + App Insights
# --------------------------------------------------
resource "azurerm_log_analytics_workspace" "law" {
  name                = "${var.aml_workspace_name}-law"
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = var.tags
}

resource "azurerm_application_insights" "appi" {
  name                = "${var.aml_workspace_name}-appi"
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  application_type    = "web"
  workspace_id        = azurerm_log_analytics_workspace.law.id
  tags                = var.tags
}

# --------------------------------------------------
# Storage Account (AML Default Datastore)
# --------------------------------------------------
resource "azurerm_storage_account" "sa" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false
  shared_access_key_enabled       = false
  default_to_oauth_authentication = true

  network_rules {
    default_action = "Deny"
    bypass         = ["AzureServices"]
  }

  tags = var.tags
}

# --------------------------------------------------
# Azure Container Registry
# --------------------------------------------------
resource "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = var.location
  sku                 = "Premium"
  admin_enabled       = false
  tags                = var.tags
}

# --------------------------------------------------
# Key Vault
# --------------------------------------------------
resource "azurerm_key_vault" "kv" {
  name                       = var.key_vault_name
  location                   = var.location
  resource_group_name        = azurerm_resource_group.rg.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  purge_protection_enabled   = true
  soft_delete_retention_days = 7

  network_acls {
    default_action = "Deny"
    bypass         = "AzureServices"
  }

  tags = var.tags
}

# --------------------------------------------------
# Azure Machine Learning Workspace (Managed VNet)
# --------------------------------------------------
resource "azurerm_machine_learning_workspace" "aml" {
  name                = var.aml_workspace_name
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name

  application_insights_id = azurerm_application_insights.appi.id
  key_vault_id            = azurerm_key_vault.kv.id
  storage_account_id      = azurerm_storage_account.sa.id
  container_registry_id   = azurerm_container_registry.acr.id

  public_network_access_enabled = false

  identity {
    type = "SystemAssigned"
  }

  managed_network {
    isolation_mode = var.aml_isolation_mode
    # Valid values commonly used: "AllowInternetOutbound" or "AllowOnlyApprovedOutbound"
  }


  tags = merge(
    var.tags,
    {
      workload = "lstm-timeseries"
      repo     = "aml-v2-lstm-ts-forecasting-demo"
    }
  )
}

# --------------------------------------------------
# AML CPU Compute Cluster
# --------------------------------------------------
resource "azurerm_machine_learning_compute_cluster" "cpu" {
  name                          = "cpu-cluster"
  location                      = var.location
  machine_learning_workspace_id = azurerm_machine_learning_workspace.aml.id
  vm_size                       = "Standard_DS3_v2"
  vm_priority                   = "Dedicated"

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 4
    scale_down_nodes_after_idle_duration = "PT120S"
  }

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}

# --------------------------------------------------
# Private DNS Zones (workspace inbound via jumpbox)
# --------------------------------------------------
resource "azurerm_private_dns_zone" "aml_api" {
  name                = "privatelink.api.azureml.ms"
  resource_group_name = azurerm_resource_group.rg.name
  tags                = var.tags
}

resource "azurerm_private_dns_zone" "aml_notebooks" {
  name                = "privatelink.notebooks.azure.net"
  resource_group_name = azurerm_resource_group.rg.name
  tags                = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "aml_api_link" {
  name                  = "aml-api-dns-link"
  resource_group_name   = azurerm_resource_group.rg.name
  private_dns_zone_name = azurerm_private_dns_zone.aml_api.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
}

resource "azurerm_private_dns_zone_virtual_network_link" "aml_notebooks_link" {
  name                  = "aml-notebooks-dns-link"
  resource_group_name   = azurerm_resource_group.rg.name
  private_dns_zone_name = azurerm_private_dns_zone.aml_notebooks.name
  virtual_network_id    = azurerm_virtual_network.vnet.id
}

# --------------------------------------------------
# Workspace Private Endpoint (inbound from jumpbox)
# --------------------------------------------------
resource "azurerm_private_endpoint" "aml_pe" {
  name                = "${var.aml_workspace_name}-pe"
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  subnet_id           = azurerm_subnet.snet_jumpbox.id

  private_service_connection {
    name                           = "${var.aml_workspace_name}-psc"
    private_connection_resource_id = azurerm_machine_learning_workspace.aml.id
    subresource_names              = ["amlworkspace"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name = "aml-dns-group"
    private_dns_zone_ids = [
      azurerm_private_dns_zone.aml_api.id,
      azurerm_private_dns_zone.aml_notebooks.id,
    ]
  }

  tags = var.tags
}