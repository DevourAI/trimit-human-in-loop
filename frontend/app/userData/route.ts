import { cookies } from 'next/headers';
import { type NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const body = await request.json();
  const userData = body.userData;
  const cookieStore = cookies();
  cookieStore.set('userData', userData);
  return NextResponse.json(
    { message: 'success' },
    {
      status: 200,
    }
  );
}

export async function GET(request: NextRequest) {
  const cookieStore = cookies();
  const userData = cookieStore.get('userData');
  if (!userData) {
    return NextResponse.json(
      { message: 'No user data found' },
      { status: 401 }
    );
  }
  return NextResponse.json(
    { userData: userData },
    {
      status: 200,
    }
  );
}
