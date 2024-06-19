'use client';
import { cva, type VariantProps } from 'class-variance-authority';
import * as React from 'react';

import { cn } from '@/lib/utils';

const headingVariants = cva('font-bold leading-tight tracking-tighter', {
  variants: {
    size: {
      default: 'text-xl md:text-2xl lg:leading-[1.2]',
      sm: 'text-sm md:text-lg lg:leading-[1.3]',
      lg: 'text-4xl md:text-6xl lg:leading-[1.0]',
    },
  },
  defaultVariants: {
    size: 'default',
  },
});

export interface HeadingProps
  extends React.HTMLAttributes<HTMLHeadingElement>,
    VariantProps<typeof headingVariants> {}

const Heading = React.forwardRef<HTMLHeadingElement, HeadingProps>(
  ({ className, size, ...props }, ref) => {
    return (
      <h1
        className={cn(headingVariants({ size, className }))}
        ref={ref}
        {...props}
      >
        {props.children}
      </h1>
    );
  }
);
Heading.displayName = 'Heading';

export { Heading, headingVariants };
