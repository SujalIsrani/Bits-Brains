import { EventData, Page } from '@nativescript/core';

let page: Page;

export function navigatingTo(args: EventData) {
  page = <Page>args.object;
}

export function onGetStarted() {
  page.frame.navigate('onboarding-page');
}