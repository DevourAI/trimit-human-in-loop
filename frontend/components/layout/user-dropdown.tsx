// Shown when the user is logged in
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuSeparator,
  DropdownMenuItem,
} from '@/components/ui/dropdown-menu';
import { Avatar, AvatarImage, AvatarFallback } from '@/components/ui/avatar';
import { Button } from '@/components/ui/button';
import { useUser } from '@/contexts/user-context';
import { ExitIcon } from '@radix-ui/react-icons';

export default function UserDropdown() {
  const { userData, logout } = useUser();

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Avatar className="h-8 w-8 cursor-pointer border border-border shadow-lg">
          <AvatarImage src={userData.picture || '/placeholder-user.jpg'} />
          <AvatarFallback>
            {userData.name ? userData.name[0] : 'JD'}
          </AvatarFallback>
        </Avatar>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-64 rounded-lg bg-background p-2 shadow-lg ring-1 ring-border">
        <div className="flex items-center justify-between py-2 px-3">
          <div className="flex items-center gap-2">
            <Avatar className="h-8 w-8 cursor-pointer">
              <AvatarImage src={userData.picture || '/placeholder-user.jpg'} />
              <AvatarFallback>
                {userData.name ? userData.name[0] : 'JD'}
              </AvatarFallback>
            </Avatar>
            <div className="leading-none overflow-hidden">
              <p className="font-medium truncate">
                {userData.name || 'John Doe'}
              </p>
              <p className="text-sm text-muted-foreground truncate">
                {userData.email || 'john@example.com'}
              </p>
            </div>
          </div>
        </div>
        <DropdownMenuSeparator />
        <DropdownMenuItem>
          <Button
            variant="ghost"
            className="w-full justify-start cursor-pointer"
            onClick={logout}
          >
            <ExitIcon className="mr-2 h-4 w-4" />
            Log out
          </Button>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
