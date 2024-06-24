'use client';

import { ArrowUpIcon } from '@radix-ui/react-icons';
import { FC, useEffect, useRef, useState } from 'react';

import { AutosizeTextarea } from '@/components/ui/autosize-textarea';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface Message {
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
  initialMessages: Message[];
  onNewMessage: (
    userMessage: string,
    callback: (aiMessage: string) => void
  ) => void;
}

const Chat: FC<ChatProps> = ({ initialMessages, onNewMessage }) => {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  useEffect(() => {
    setMessages(initialMessages);
  }, [initialMessages]);
  console.log('initialMessages inside Chat', initialMessages);
  console.log('messages inside Chat', messages);
  const [inputValue, setInputValue] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = () => {
    if (inputValue.trim() === '') return;

    const newMessage: Message = { sender: 'Human', text: inputValue };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setInputValue('');

    onNewMessage(newMessage.text, (aiMessage: string) => {
      setMessages((prevMessages) => [
        ...prevMessages,
        { sender: 'AI', text: aiMessage },
      ]);
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSendMessage();
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
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <div className="flex-1 relative">
          <AutosizeTextarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message..."
            className="w-full pr-12 h-auto max-h-48 resize-none"
          />

          <Button
            type="submit"
            className="absolute right-1 bottom-1 rounded-full"
            size="sm"
            variant="secondary"
          >
            Retry
            <ArrowUpIcon />
          </Button>
        </div>
      </form>
    </div>
  );
};
export default Chat;
