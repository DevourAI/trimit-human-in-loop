import { google } from 'googleapis';
import { type NextRequest, NextResponse } from 'next/server';

function videoIdFromWeblink(url: string): string | null {
  // Regular expression to match different YouTube URL formats
  const regex =
    /(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})/;

  // Match the URL against the regular expression
  const match = url.match(regex);

  // Return the video ID if a match is found
  return match ? match[1] : null;
}

// This function initializes the Google API client and fetches the video title
async function fetchYouTubeVideoTitle(weblink: string): Promise<string | null> {
  const videoId = videoIdFromWeblink(weblink);

  if (!videoId) {
    return null;
  }
  const youtube = google.youtube({
    version: 'v3',
    auth: process.env.GOOGLE_API_KEY,
  });

  const params = {
    id: [videoId],
    part: ['snippet'],
  };
  try {
    // Load the client and set the API key
    const res = await youtube.videos.list(params);
    if (!res || !res.data || !res.data.items || res.data.items.length === 0) {
      return null;
    }
    const videoData = res.data.items[0];
    if (!videoData || !videoData.snippet) {
      return null;
    }
    return videoData.snippet.title || null;
  } catch (error) {
    console.error('Error: ', error);
  }
  return null;
}

function extractWeblink(req: NextRequest): string | null {
  // Access the URL object from the request
  const url = req.nextUrl;

  // Retrieve the 'weblink' query parameter
  const weblink = url.searchParams.get('weblink');

  return weblink;
}

export async function GET(request: NextRequest) {
  const weblink = extractWeblink(request);
  if (!weblink) {
    return NextResponse.json(
      { error: 'No weblink provided' },
      {
        status: 400,
      }
    );
  }
  const title = await fetchYouTubeVideoTitle(weblink);
  return NextResponse.json(
    { title: title },
    {
      status: 200,
    }
  );
}
