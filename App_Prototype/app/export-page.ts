import { EventData, Page, Utils, isAndroid, isIOS, Application } from '@nativescript/core';
import { alert, confirm } from '@nativescript/core/ui/dialogs';
import { DataStorage } from './services/data-storage';
declare const android: any;

let page: Page;

export function navigatingTo(args: EventData) {
    page = <Page>args.object;
}

// Helper function for browser download
function downloadFileBrowser(content: string, fileName: string, mimeType: string) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

export async function onExportCSV() {
    try {
        const result = await DataStorage.exportToCSV();
        
        // Check if running in browser (StackBlitz)
        if (typeof document !== 'undefined' && !isAndroid && !isIOS) {
            // Generate CSV content from the available data
            const csvContent = generateCSVContent(result);
            downloadFileBrowser(csvContent, result.fileName, 'text/csv');
            alert({
                title: 'Export Successful',
                message: `CSV file with ${result.sessionCount} sessions has been downloaded.`,
                okButtonText: 'OK'
            });
            return;
        }

        // Native implementation
        const openLocation = await confirm({
            title: 'Export Successful',
            message: `CSV file created with ${result.sessionCount} sessions.\n\nFile saved to: ${result.fileName}\n\nWould you like to open the file location?`,
            okButtonText: 'Open Location',
            cancelButtonText: 'OK'
        });

        if (openLocation) {
            try {
                if (isAndroid) {
                    // Correct way to access Android Intent class
                    const Intent = (<any>android).content.Intent;
                    const intent = new Intent(Intent.ACTION_VIEW);
                    intent.setDataAndType(android.net.Uri.parse(result.filePath), "*/*");
                    intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                    
                    const context = Utils.android.getApplicationContext();
                    context.startActivity(Intent.createChooser(intent, "Open File Manager"));
                } else if (isIOS) {
                    alert({
                        title: 'File Location',
                        message: `File saved at: ${result.filePath}\n\nYou can access this file through the Files app.`,
                        okButtonText: 'OK'
                    });
                }
            } catch (error) {
                console.error('Error opening file manager:', error);
                alert({
                    title: 'File Saved',
                    message: `CSV file saved successfully!\n\nLocation: ${result.filePath}\n\nYou can find this file in your device's Downloads folder.`,
                    okButtonText: 'OK'
                });
            }
        }
        
    } catch (error) {
        alert({
            title: 'Export Failed',
            message: error.message,
            okButtonText: 'OK'
        });
    }
}

// Helper function to generate CSV content
function generateCSVContent(data: { filePath: string; fileName: string; sessionCount: number }): string {
    // Implement your CSV generation logic here
    const headers = "SessionID,Date,Duration,Accuracy\n";
    const rows = Array(data.sessionCount).fill("1,2023-01-01,60,95%").join("\n");
    return headers + rows;
}

export function onBack() {
    page.frame.goBack();
}