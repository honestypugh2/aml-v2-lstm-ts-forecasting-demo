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
# Log Analytics + Application Insights
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
# Storage Account (AML default datastore)
# --------------------------------------------------
resource "azurerm_storage_account" "sa" {
  name                     = var.storage_account_name
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  min_tls_version                 = "TLS1_2"
  allow_nested_items_to_be_public = false

  tags = var.tags
}

# --------------------------------------------------
# Azure Container Registry
# --------------------------------------------------
resource "azurerm_container_registry" "acr" {
  name                = var.acr_name
  resource_group_name = azurerm_resource_group.rg.name
  location            = var.location
  sku                 = "Basic"
  admin_enabled       = false

  tags = var.tags
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
  purge_protection_enabled   = false
  soft_delete_retention_days = 7

  tags = var.tags
}

# --------------------------------------------------
# Azure Machine Learning Workspace (Public Access)
# --------------------------------------------------
resource "azurerm_machine_learning_workspace" "aml" {
  name                = var.aml_workspace_name
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name

  application_insights_id = azurerm_application_insights.appi.id
  key_vault_id            = azurerm_key_vault.kv.id
  storage_account_id      = azurerm_storage_account.sa.id
  container_registry_id   = azurerm_container_registry.acr.id

  public_network_access_enabled = true

  identity {
    type = "SystemAssigned"
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
# Azure ML Compute Instance (for notebooks / dev)
# --------------------------------------------------
resource "azurerm_machine_learning_compute_instance" "ci" {
  name                          = "${var.compute_instance_name}-${substr(md5(azurerm_machine_learning_workspace.aml.id), 0, 6)}"
  machine_learning_workspace_id = azurerm_machine_learning_workspace.aml.id
  virtual_machine_size          = var.compute_vm_size

  authorization_type = "personal"

  assign_to_user {
    object_id = data.azurerm_client_config.current.object_id
    tenant_id = data.azurerm_client_config.current.tenant_id
  }

  tags = var.tags
}

# --------------------------------------------------
# Azure ML Compute Cluster (for training jobs)
# --------------------------------------------------
resource "azurerm_machine_learning_compute_cluster" "cc" {
  name                          = var.compute_cluster_name
  machine_learning_workspace_id = azurerm_machine_learning_workspace.aml.id
  vm_size                       = var.compute_vm_size
  vm_priority                   = "Dedicated"
  location                      = var.location

  scale_settings {
    min_node_count                       = 0
    max_node_count                       = 2
    scale_down_nodes_after_idle_duration = "PT120S"
  }

  identity {
    type = "SystemAssigned"
  }

  tags = var.tags
}
