import Link from 'next/link';
import React from 'react';

import HeaderNav from '@/components/layout/header-nav';
import { ThemeToggle } from '@/components/layout/theme-toggle';
import Login from '@/components/login';

interface HeaderProps {
  children?: React.ReactNode;
}

export const Logo = () => {
  return (
    <Link href="/">
      <div className="font-bold leading-tight tracking-tighter cursor-pointer">
        TrimIt
      </div>
    </Link>
  );
};

export default function Header({ children }: HeaderProps) {
  return (
    <header className="bg-background/90 sticky top-0 z-50 backdrop-filter backdrop-blur-lg w-full">
      <div className="w-full border-b h-16 flex items-center">
        <div className="w-full flex justify-between items-center md:px-8">
          <div className="flex items-center gap-4 md:gap-8">
            <Logo />
            <div className="hidden md:block">
              <HeaderNav />
            </div>
          </div>
          {children}

          <div className="flex items-center gap-2 md:gap-3">
            <ThemeToggle />
            <Login />
          </div>
        </div>
      </div>
    </header>
  );
}
