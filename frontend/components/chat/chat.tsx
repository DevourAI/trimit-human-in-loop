'use client';

import { ArrowUpIcon } from '@radix-ui/react-icons';
import { useEffect, useRef, useState } from 'react';

import { AutosizeTextarea } from '@/components/ui/autosize-textarea';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface Message {
  sender: 'human' | 'llm';
  text: string;
}

function ChatMessage({ message }: { message: Message }) {
  return (
    <div
      className={cn(
        'p-2 rounded-full',
        message.sender === 'human' ? 'ml-auto bg-muted w-fit' : ''
      )}
    >
      {message.text}
    </div>
  );
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([
    { sender: 'llm', text: 'Hello! How can I assist you today?' },
    { sender: 'human', text: 'Can you tell me a joke?' },
    {
      sender: 'llm',
      text: 'Sure! Why dont scientists trust atoms? Because they make up everything!',
    },
  ]);
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

    const newMessage: Message = { sender: 'human', text: inputValue };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    setInputValue('');

    // Simulate LLM response
    setTimeout(() => {
      const llmResponse: Message = {
        sender: 'llm',
        text: 'This is a response from the LLM.',
      };
      setMessages((prevMessages) => [...prevMessages, llmResponse]);
    }, 1000);
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
        {messages.map((message, index) => (
          <ChatMessage key={index} message={message} />
        ))}
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
            size="icon"
            variant="secondary"
          >
            <ArrowUpIcon />
          </Button>
        </div>
      </form>
    </div>
  );
}
