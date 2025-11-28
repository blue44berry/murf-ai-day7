'use client';

import { RoomAudioRenderer, StartAudio } from '@livekit/components-react';
import type { AppConfig } from '@/app-config';
import { SessionProvider } from '@/components/app/session-provider';
import { ViewController } from '@/components/app/view-controller';
import { Toaster } from '@/components/livekit/toaster';

interface AppProps {
  appConfig: AppConfig;
}

export function App({ appConfig }: AppProps) {
  return (
    <SessionProvider appConfig={appConfig}>
      {/* Make the main content hug the bottom */}
      <main className="flex min-h-screen flex-col justify-end items-center pb-10">
        <div className="w-full max-w-4xl">
          <ViewController />
        </div>
      </main>

      {/* Fix the talk button at the bottom center */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-20">
        <StartAudio label="Talk to our AI barista" />
      </div>

      <RoomAudioRenderer />
      <Toaster />
    </SessionProvider>
  );
}
