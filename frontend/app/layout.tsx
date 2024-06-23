import '../styles/globals.css';

import { GoogleOAuthProvider } from '@react-oauth/google';
import { Inter as FontSans } from 'next/font/google';

import { Toaster } from '@/components/ui/toaster';
import { UserProvider } from '@/contexts/user-context';
import { cn } from '@/lib/utils';
import { ThemeProvider } from '@/providers/theme-provider';

const clientId = process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID;

const fontSans = FontSans({
  subsets: ['latin'],
  variable: '--font-sans',
});

export const metadata = {
  title: 'TrimIt Interview Builder',
  description: 'Raw interview footage to edited video, no timeline required.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head />
      <body
        className={
          cn(
            'min-h-screen bg-background font-sans antialiased',
            fontSans.variable
          ) ?? ''
        }
      >
        <Toaster />
        <GoogleOAuthProvider clientId={clientId ?? ''}>
          <UserProvider>
            <ThemeProvider
              attribute="class"
              defaultTheme="dark"
              enableSystem
              disableTransitionOnChange
            >
              {children}
              <Toaster />
            </ThemeProvider>
          </UserProvider>
        </GoogleOAuthProvider>
      </body>
    </html>
  );
}
