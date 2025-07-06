import { EventData, Page, TextField, Button, Switch } from '@nativescript/core';
import { DataStorage } from './services/data-storage';
import { UserMetadata } from './models/typing-session';

let page: Page;
let patientIdField: TextField;
let updrsScoreField: TextField;
let tremorSeverityField: TextField;
let parkinsonsSwitch: Switch;
let medicationStatusField: TextField;

export function navigatingTo(args: EventData) {
  page = <Page>args.object;
  
  patientIdField = page.getViewById('patientId') as TextField;
  updrsScoreField = page.getViewById('updrsScore') as TextField;
  tremorSeverityField = page.getViewById('tremorSeverity') as TextField;
  parkinsonsSwitch = page.getViewById('parkinsonsSwitch') as Switch;
  medicationStatusField = page.getViewById('medicationStatus') as TextField;

  // Load existing metadata if available
  const existingMetadata = DataStorage.getUserMetadata();
  if (existingMetadata) {
    patientIdField.text = existingMetadata.patientId;
    updrsScoreField.text = existingMetadata.updrsScore.toString();
    tremorSeverityField.text = existingMetadata.tremorSeverity.toString();
    parkinsonsSwitch.checked = existingMetadata.diagnosis === 'Parkinson\'s';
    medicationStatusField.text = existingMetadata.medicationStatus;
  }
}

export function onContinue() {
  const patientId = patientIdField.text?.trim();
  const updrsScore = parseFloat(updrsScoreField.text);
  const tremorSeverity = parseFloat(tremorSeverityField.text);
  const diagnosis = parkinsonsSwitch.checked ? 'Parkinson\'s' : 'Healthy';
  const medicationStatus = medicationStatusField.text?.trim() || 'NA';

  // Validation
  if (!patientId) {
    alert('Please enter a Patient ID');
    return;
  }

  if (isNaN(updrsScore) || updrsScore < 0 || updrsScore > 132) {
    alert('Please enter a valid UPDRS score (0-132)');
    return;
  }

  if (isNaN(tremorSeverity) || tremorSeverity < 0.1 || tremorSeverity > 2.0) {
    alert('Please enter a valid tremor severity (0.1-2.0)');
    return;
  }

  if (!['On', 'Off', 'NA'].includes(medicationStatus)) {
    alert('Please enter a valid medication status (On/Off/NA)');
    return;
  }

  const metadata: UserMetadata = {
    patientId,
    diagnosis: diagnosis as 'Parkinson\'s' | 'Healthy',
    updrsScore,
    tremorSeverity,
    medicationStatus: medicationStatus as 'On' | 'Off' | 'NA'
  };

  DataStorage.saveUserMetadata(metadata);
  
  // Navigate to main app
  page.frame.navigate({
    moduleName: 'main-page',
    clearHistory: true
  });
}