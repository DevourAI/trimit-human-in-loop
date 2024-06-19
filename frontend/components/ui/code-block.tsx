import {
  ChevronDownIcon,
  ChevronUpIcon,
  ClipboardIcon,
} from '@radix-ui/react-icons';
import React, { useRef, useState } from 'react';

import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface CodeBlockProps {
  code: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code }) => {
  const ref = useRef<HTMLElement>(null);
  const [expanded, setExpanded] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
  };

  return (
    <div>
      <pre className={cn('p-2 pb-0 rounded-md shadow-md bg-muted', {})}>
        <code
          ref={ref}
          className={cn(
            'p-1 rounded-md text-muted-foreground text-xs whitespace-pre-wrap',
            { 'max-h-20 overflow-hidden': !expanded }
          )}
        >
          {code}
        </code>
        <div className="flex space-x-2 mt-1 justify-end">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? (
              <ChevronUpIcon className="mr-2" />
            ) : (
              <ChevronDownIcon className="mr-2" />
            )}
            {expanded ? 'Show Less' : 'Show More'}
          </Button>
          <Button variant="ghost" size="sm" onClick={copyToClipboard}>
            <ClipboardIcon className="mr-2" />
            Copy
          </Button>
        </div>
      </pre>
    </div>
  );
};

export default CodeBlock;
