'use client';
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuIndicator,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  NavigationMenuViewport,
  navigationMenuTriggerStyle,
} from '@/components/ui/navigation-menu';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';

const links = [
  //   {
  //     label: "Home",
  //     href: "/",
  //   },
  {
    label: 'Builder',
    href: '/builder',
  },
  {
    label: 'Videos',
    href: '/videos',
  },
  {
    label: 'Projects',
    href: '/projects',
  },
];

export default function HeaderNav() {
  const pathname = usePathname();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return null;
  }

  return (
    <NavigationMenu>
      <NavigationMenuList>
        {links.map((link) => (
          <NavigationMenuItem key={link.href}>
            <Link href={link.href} passHref>
              <NavigationMenuLink
                className={`${navigationMenuTriggerStyle()} ${
                  pathname === link.href ? 'bg-muted' : ''
                }`}
              >
                {link.label}
              </NavigationMenuLink>
            </Link>
          </NavigationMenuItem>
        ))}
      </NavigationMenuList>
    </NavigationMenu>
  );
}
