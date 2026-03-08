resource "azurerm_user_assigned_identity" "jumpbox_identity" {
  name                = "${var.aml_workspace_name}-jumpbox-identity"
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  tags                = var.tags
}

resource "azurerm_network_interface" "jumpbox_nic" {
  name                = "${var.aml_workspace_name}-jumpbox-nic"
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name

  ip_configuration {
    name                          = "internal"
    subnet_id                     = azurerm_subnet.snet_jumpbox.id
    private_ip_address_allocation = "Dynamic"
  }
}

resource "azurerm_windows_virtual_machine" "jumpbox" {
  name                = "${var.aml_workspace_name}-jumpbox"
  computer_name       = "aml-jumpbox"
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  size                = "Standard_D2s_v3"

  admin_username = "azureuser"
  admin_password = var.admin_password

  network_interface_ids = [
    azurerm_network_interface.jumpbox_nic.id
  ]

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.jumpbox_identity.id]
  }

  source_image_reference {
    publisher = "MicrosoftWindowsServer"
    offer     = "WindowsServer"
    sku       = "2022-datacenter-azure-edition"
    version   = "latest"
  }

  tags = var.tags
}

# Grant the jumpbox managed identity Contributor on the resource group
resource "azurerm_role_assignment" "jumpbox_contributor" {
  scope                = azurerm_resource_group.rg.id
  role_definition_name = "Contributor"
  principal_id         = azurerm_user_assigned_identity.jumpbox_identity.principal_id
}

# Grant the jumpbox managed identity AzureML Data Scientist role on the workspace
resource "azurerm_role_assignment" "jumpbox_aml_ds" {
  scope                = azurerm_machine_learning_workspace.aml.id
  role_definition_name = "AzureML Data Scientist"
  principal_id         = azurerm_user_assigned_identity.jumpbox_identity.principal_id
}