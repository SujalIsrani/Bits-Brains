# NeuroType Keyboard - Setup Guide

This guide explains how to run the **NeuroType Keyboard** app on **Android Studio** or **VS Code** for testing and development.

---

## Prerequisites

Before running the app, ensure you have the following installed:

- [Node.js](https://nodejs.org/) (v16.x or later)
- [NativeScript CLI](https://docs.nativescript.org/setup/)
  ```bash
  npm install -g nativescript
  ```
- **Android Studio** (for Android emulator)
  - Install via [official site](https://developer.android.com/studio)
  - Configure an **Android Virtual Device (AVD)**
- **VS Code** (optional, for editing)
  - Install the [NativeScript extension](https://marketplace.visualstudio.com/items?itemName=NativeScript.nativescript)

---

## 1. Running the App in VS Code (Preview Mode)

### Step 1: Clone the Repository

```bash
git clone YOUR_REPO_URL
cd YOUR_PROJECT_FOLDER
```

### Step 2: Install Dependencies

```bash
npm install
```

### Step 3: Start NativeScript Preview

```bash
ns preview
```

- A **QR code** will appear in the terminal.
- Scan it using the **NativeScript Playground App** ([Android](https://play.google.com/store/apps/details?id=org.nativescript.play) / [iOS](https://apps.apple.com/us/app/nativescript-playground/id1263543946)).

---

## 2. Running in Android Studio (Full Emulator/Device)

### Step 1: Build the Android Project

```bash
ns build android
```

### Step 2: Open in Android Studio

1. Launch **Android Studio**
2. Select **"Open an Existing Project"**
3. Navigate to:
   ```
   YOUR_PROJECT_FOLDER/platforms/android
   ```

### Step 3: Run on Emulator/Device

1. Ensure an **Android Virtual Device (AVD)** is running.
2. Click **"Run" (▶️)** in Android Studio.
3. The app will launch on the emulator/connected device.

---

## Troubleshooting

| Issue                     | Solution                                                                 |
|--------------------------|--------------------------------------------------------------------------|
| `ns preview` fails       | Ensure `node_modules` is installed (`npm install`).                      |
| Android build errors     | Check `platforms/android` exists (`ns build android`).                   |
| Emulator not detected    | Run `ns run android --device DEVICE_ID` (list devices with `ns device android --available-devices`). |

---

## Judges' Quick Test

If judges only need to **preview** (not full build):

1. `git clone YOUR_REPO`
2. `npm install`
3. `ns preview` → Scan QR with **NativeScript Playground App**

---

## Need Help?

- **NativeScript Docs**: [https://docs.nativescript.org/](https://docs.nativescript.org/)
- **VS Code Setup**: [https://marketplace.visualstudio.com/items?itemName=NativeScript.nativescript](https://marketplace.visualstudio.com/items?itemName=NativeScript.nativescript)

---

🚀 **Happy Testing!** Let us know if you need further assistance.