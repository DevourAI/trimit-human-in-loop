import React from 'react';

import Header from '@/components/layout/header';

export default function AppShell({
  children,
  title = '',
  subtitle = '',
  actions,
}: {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="container min-h-screen justify-center items-center w-full">
      <Header />
      <main className="w-full py-4 md:p-8">
        <div className="mb-4 flex justify-between items-center">
          <div>
            {title && (
              <h1 className="text-2xl font-bold leading-tight tracking-tighter block mb-2">
                {title}
              </h1>
            )}
            {subtitle && (
              <h2 className="text-muted-foreground text-sm block mb-2">
                {subtitle}
              </h2>
            )}
          </div>
          {actions && <div>{actions}</div>}
        </div>
        {children}
      </main>
    </div>
  );
}
