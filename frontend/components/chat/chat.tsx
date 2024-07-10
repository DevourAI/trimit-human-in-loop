'use client';

import { FC, useEffect, useRef } from 'react';

import { AutosizeTextarea } from '@/components/ui/autosize-textarea';
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
  isNewStep: boolean;
  onChange: (userMessage: string) => void;
}

const Chat: FC<ChatProps> = ({
  isNewStep,
  messages,
  userMessage,
  onSubmit,
  onChange,
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
            placeholder="Type your message..."
            className="w-full pr-12 h-auto max-h-48 resize-none"
          />
        </div>
      </form>
    </div>
  );
};
export default Chat;
