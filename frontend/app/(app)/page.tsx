import { headers } from 'next/headers';
import { App } from '@/components/app/app';
import { getAppConfig } from '@/lib/utils';

export default async function Page() {
  const hdrs = await headers();
  const appConfig = await getAppConfig(hdrs);

  return (
    <div className="min-h-screen w-full bg-[#050016] flex flex-col">
      {/* Just the app, no background image */}
      <App appConfig={appConfig} />
    </div>
  );
}
