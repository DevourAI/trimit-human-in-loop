import { cookies } from 'next/headers'
import { type NextRequest } from 'next/server'

export async function POST(request: NextRequest) {
  const body = await request.json()
  const userData = body.userData
  const cookieStore = cookies()
  cookieStore.set('userData', userData)
  return Response.json({'message': 'success'}, {
    status: 200,
  })
}

export async function GET(request: NextRequest) {
  const cookieStore = cookies()
  const userData = cookieStore.get('userData')
  if (!userData) {
    return Response.json({'message': 'No user data found'}, { status: 401 })
  }
  return Response.json({'userData': userData,}, {
    status: 200,
  })
}
