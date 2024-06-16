// components/Login.tsx
'use client';
import { GoogleLogin } from '@react-oauth/google';
import { jwtDecode } from 'jwt-decode';

import UserDropdown from '@/components/layout/user-dropdown';
import { useUser } from '@/contexts/user-context';

export default function Login() {
  const { userData, login, isLoggedIn } = useUser();

  return (
    <div className="flex gap-3 items-center">
      {isLoggedIn ? (
        <UserDropdown />
      ) : (
        <GoogleLogin
          onSuccess={(credentialResponse) => {
            login(credentialResponse.credential);

            fetch(`/api/userData`, {
              body: JSON.stringify({
                userData: jwtDecode(credentialResponse.credential),
              }),
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
            }).then((res) => {});
          }}
          onError={() => {
            console.log('Login Failed');
          }}
          useOneTap
        />
      )}
    </div>
  );
}
