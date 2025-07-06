import { ApplicationSettings, Utils } from '@nativescript/core';
import { File, Folder, knownFolders, path } from '@nativescript/core';
import { TypingSession, UserMetadata } from '../models/typing-session';

export class DataStorage {
  private static readonly SESSIONS_KEY = 'typing_sessions';
  private static readonly USER_METADATA_KEY = 'user_metadata';

  static saveUserMetadata(metadata: UserMetadata): void {
    ApplicationSettings.setString(this.USER_METADATA_KEY, JSON.stringify(metadata));
  }

  static getUserMetadata(): UserMetadata | null {
    const metadataString = ApplicationSettings.getString(this.USER_METADATA_KEY);
    if (metadataString) {
      return JSON.parse(metadataString);
    }
    return null;
  }

  static saveSession(session: TypingSession): void {
    const sessions = this.getAllSessions();
    sessions.push(session);
    ApplicationSettings.setString(this.SESSIONS_KEY, JSON.stringify(sessions));
  }

  static getAllSessions(): TypingSession[] {
    const sessionsString = ApplicationSettings.getString(this.SESSIONS_KEY);
    if (sessionsString) {
      return JSON.parse(sessionsString);
    }
    return [];
  }

  private static getDownloadsFolder(): Folder {
    try {
      // For WebContainer/preview, use documents folder which is more reliable
      return knownFolders.documents();
    } catch (error) {
      console.log('Could not access Downloads folder, using Documents:', error);
      return knownFolders.documents();
    }
  }

  static async exportToCSV(): Promise<{filePath: string, fileName: string, sessionCount: number}> {
    const sessions = this.getAllSessions();
    if (sessions.length === 0) {
      throw new Error('No sessions to export');
    }

    // Create comprehensive CSV header with all metrics and timestamp
    const csvHeader = [
      'sessionId',
      'patientId', 
      'diagnosis',
      'updrsScore',
      'tremorSeverity',
      'medicationStatus',
      'timestamp',
      'sessionDate',
      'sessionTime',
      'sessionDuration',
      'totalKeystrokes',
      'typingSpeed',
      'meanHoldTime',
      'stdHoldTime',
      'holdTimeVariability',
      'meanFlightTime',
      'stdFlightTime',
      'backspaceRate',
      'pauseFrequency'
    ].join(',');

    const csvRows = sessions.map(session => {
      // Parse timestamp and create readable date/time
      const timestamp = session.timestamp.replace(/-/g, ':');
      const sessionDate = new Date(timestamp);
      const dateStr = sessionDate.toLocaleDateString();
      const timeStr = sessionDate.toLocaleTimeString();

      return [
        `"${session.sessionId}"`,
        `"${session.patientId}"`,
        `"${session.diagnosis}"`,
        session.updrsScore,
        session.tremorSeverity,
        `"${session.medicationStatus}"`,
        `"${session.timestamp}"`,
        `"${dateStr}"`,
        `"${timeStr}"`,
        session.sessionDuration.toFixed(2),
        session.totalKeystrokes,
        session.typingSpeed.toFixed(2),
        session.meanHoldTime.toFixed(2),
        session.stdHoldTime.toFixed(2),
        session.holdTimeVariability.toFixed(4),
        session.meanFlightTime.toFixed(2),
        session.stdFlightTime.toFixed(2),
        session.backspaceRate.toFixed(2),
        session.pauseFrequency.toFixed(2)
      ].join(',');
    });

    const csvContent = [csvHeader, ...csvRows].join('\n');

    // Create filename with current timestamp
    const now = new Date();
    const dateTimeStamp = now.toISOString().replace(/[:.]/g, '-').split('T')[0] + '_' + 
                         now.toTimeString().split(' ')[0].replace(/:/g, '-');
    const fileName = `NeuroTypeData_${dateTimeStamp}.csv`;

    // Save to Documents folder (more reliable in WebContainer)
    const targetFolder = this.getDownloadsFolder();
    const filePath = path.join(targetFolder.path, fileName);
    const file = File.fromPath(filePath);
    
    await file.writeText(csvContent);
    
    console.log(`CSV file saved to: ${filePath}`);
    console.log(`Total sessions exported: ${sessions.length}`);
    
    return {
      filePath: filePath,
      fileName: fileName,
      sessionCount: sessions.length
    };
  }

  static async exportToJSON(): Promise<{filePath: string, fileName: string, sessionCount: number}> {
    const sessions = this.getAllSessions();
    if (sessions.length === 0) {
      throw new Error('No sessions to export');
    }

    // Add readable timestamps to JSON export
    const enhancedSessions = sessions.map(session => ({
      ...session,
      readableTimestamp: new Date(session.timestamp.replace(/-/g, ':')).toLocaleString(),
      exportDate: new Date().toISOString()
    }));

    const exportData = {
      exportInfo: {
        exportDate: new Date().toISOString(),
        totalSessions: sessions.length,
        appVersion: '1.0.0',
        patientInfo: this.getUserMetadata()
      },
      sessions: enhancedSessions
    };

    const jsonContent = JSON.stringify(exportData, null, 2);

    // Create filename with current timestamp
    const now = new Date();
    const dateTimeStamp = now.toISOString().replace(/[:.]/g, '-').split('T')[0] + '_' + 
                         now.toTimeString().split(' ')[0].replace(/:/g, '-');
    const fileName = `NeuroTypeData_${dateTimeStamp}.json`;

    // Save to Documents folder (more reliable in WebContainer)
    const targetFolder = this.getDownloadsFolder();
    const filePath = path.join(targetFolder.path, fileName);
    const file = File.fromPath(filePath);
    
    await file.writeText(jsonContent);
    
    console.log(`JSON file saved to: ${filePath}`);
    console.log(`Total sessions exported: ${sessions.length}`);
    
    return {
      filePath: filePath,
      fileName: fileName,
      sessionCount: sessions.length
    };
  }

  static clearAllData(): void {
    ApplicationSettings.remove(this.SESSIONS_KEY);
    ApplicationSettings.remove(this.USER_METADATA_KEY);
  }
}