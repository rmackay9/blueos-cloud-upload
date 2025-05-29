# Dropbox Setup Guide

## What you're looking for in the Dropbox App Console:

### 1. App Creation
```
https://www.dropbox.com/developers/apps
┌─────────────────────────────────────┐
│ Create app                          │
│ ┌─────────────────────────────────┐ │
│ │ ○ Scoped access (recommended)   │ │
│ │ ○ App folder                    │ │
│ │ ● Full Dropbox                  │ │ ← Choose this
│ └─────────────────────────────────┘ │
│                                     │
│ App name: [BlueOS Cloud Upload]     │
└─────────────────────────────────────┘
```

### 2. Permissions Tab
```
┌─────────────────────────────────────┐
│ Permissions                         │
│                                     │
│ ☑ files.content.read               │ ← Check this
│ ☑ files.content.write              │ ← Check this
│ ☑ files.metadata.write             │ ← Check this
│ ☐ files.metadata.read              │
│ ☐ sharing.read                     │
│                                     │
│ [Submit]                           │ ← Click Submit
└─────────────────────────────────────┘
```

### 3. Settings Tab - OAuth 2 Section
```
┌─────────────────────────────────────┐
│ Settings                            │
│                                     │
│ OAuth 2                            │
│ ┌─────────────────────────────────┐ │
│ │ Generated access token          │ │
│ │                                 │ │
│ │ [Generate access token]         │ │ ← Click this
│ │                                 │ │
│ │ sl.xxxxxxxxxxxxxxxxxxxxxxxx     │ │ ← Copy this token
│ │ xxxxxxxxxxxxxxxxxxxxxxxxxxxx    │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

## What goes in the Cloud Upload Extension:

### Username/Email Field:
```
┌─────────────────────────────────────┐
│ Username/Email: [your@email.com]    │ ← Optional (for display only)
│                 [              ]    │ ← Or leave blank
└─────────────────────────────────────┘
```

### Password/API Key Field:
```
┌─────────────────────────────────────┐
│ Password/API Key:                   │
│ [sl.xxxxxxxxxxxxxxxxxxxxxxxx]      │ ← Paste the ACCESS TOKEN here
│                                     │   (NOT your Dropbox password)
└─────────────────────────────────────┘
```

## Common Mistakes:
❌ Using your Dropbox account password
❌ Using the App key or App secret
❌ Not setting the required permissions
❌ Copying only part of the access token

✅ Use the FULL access token from OAuth 2 section
✅ Set the three required permissions
✅ Username field is optional for Dropbox
