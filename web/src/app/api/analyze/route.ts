import { NextResponse } from 'next/server';

// Proxy to the Python segmentation service. The browser never talks to it
// directly: that keeps ai-api off the public network, avoids CORS, and leaves
// one place to add auth and rate limiting later.
export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const AI_API_URL = process.env.AI_API_URL ?? 'http://localhost:8010';
const TIMEOUT_MS = 60_000;

export async function POST(request: Request) {
  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return NextResponse.json({ error: 'รูปแบบคำขอไม่ถูกต้อง' }, { status: 400 });
  }

  if (!form.get('file')) {
    return NextResponse.json({ error: 'ไม่พบไฟล์ภาพในคำขอ' }, { status: 400 });
  }

  try {
    const res = await fetch(`${AI_API_URL}/api/v1/segment`, {
      method: 'POST',
      body: form,
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });

    const json = await res.json().catch(() => null);

    if (!res.ok) {
      // FastAPI puts our Thai-language message in `detail`.
      const message =
        (json && (json.detail ?? json.error)) || 'วิเคราะห์ภาพไม่สำเร็จ';
      return NextResponse.json({ error: message }, { status: res.status });
    }

    return NextResponse.json(json);
  } catch (err: any) {
    const timedOut = err?.name === 'TimeoutError';
    console.error('[analyze] ai-api request failed:', err?.message ?? err);
    return NextResponse.json(
      {
        error: timedOut
          ? 'ระบบ AI ใช้เวลานานเกินไป กรุณาลองใหม่'
          : 'เชื่อมต่อระบบ AI ไม่ได้ กรุณาตรวจสอบว่า ai-api ทำงานอยู่',
      },
      { status: timedOut ? 504 : 502 },
    );
  }
}
