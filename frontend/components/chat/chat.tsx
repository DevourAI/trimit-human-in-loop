'use client';

import { ArrowUpIcon } from '@radix-ui/react-icons';
import { FC, useEffect, useRef } from 'react';

import { AutosizeTextarea } from '@/components/ui/autosize-textarea';
import { Button } from '@/components/ui/button';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';

export interface Message {
  sender: 'Human' | 'AI';
  text: string;
}

function ChatMessage({ message }: { message: Message }) {
  return (
    <div
      className={cn(
        'p-2 rounded-full',
        message.sender === 'Human' ? 'ml-auto bg-muted w-fit' : ''
      )}
    >
      {message.text}
    </div>
  );
}

interface ChatProps {
  messages: Message[];
  userMessage: string;
  onSubmit: (options?: any) => void;
  onChange: (userMessage: string) => void;
  isLoading: boolean;
  disabled?: boolean;
  disabledMessage?: string;
}

const Chat: FC<ChatProps> = ({
  disabled,
  disabledMessage,
  messages,
  userMessage,
  onSubmit,
  onChange,
  isLoading,
}) => {
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = () => {
    onSubmit();
    if (userMessage.trim() === '') {
      return;
    }
    onChange('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const generateButton = (
    <Button
      variant="default"
      onClick={onSubmit}
      disabled={isLoading || disabled}
    >
      <ArrowUpIcon className="mr-2" />
      Generate video
    </Button>
  );

  return (
    <div className="w-full">
      <div className="mb-6 max-h-48 overflow-y-auto space-y-4">
        {messages
          ? messages.map((message, index) => (
              <ChatMessage key={index} message={message} />
            ))
          : null}
        <div ref={messagesEndRef} />
      </div>
      <form className="flex space-x-2">
        <div className="flex-1 relative">
          <AutosizeTextarea
            value={userMessage}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Describe what you want to create..."
            className="w-full pr-12 h-auto max-h-48 resize-none"
          />
        </div>
      </form>

      {disabled ? (
        <TooltipProvider>
          <Tooltip delayDuration={0}>
            <TooltipTrigger asChild>
              <div className="inline-block">{generateButton}</div>
            </TooltipTrigger>
            <TooltipContent>
              <p>{disabledMessage}</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      ) : (
        generateButton
      )}
    </div>
  );
};
export default Chat;
