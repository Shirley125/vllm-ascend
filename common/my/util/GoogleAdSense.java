package my.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.net.URL;
import java.net.URLEncoder;

import javax.servlet.http.HttpServletRequest;

/**
 * 嵌入Google AdSense 广告代码
 * 
 * @author Winter Lau
 * @date 2009-7-1 下午10:49:27
 */
public class GoogleAdSense {

	private static final String PAGEAD = "http://pagead2.googlesyndication.com/pagead/ads?";

	static void googleAppendUrl(StringBuilder url, String param, String value)
	    throws UnsupportedEncodingException {
	  if (value != null) {
	    String encodedValue = URLEncoder.encode(value, "UTF-8");
	    url.append("&").append(param).append("=").append(encodedValue);
	  }
	}

	static void googleAppendScreenRes(StringBuilder url, String uaPixels,
	    String xUpDevcapScreenpixels) {
	  String screenRes = uaPixels;
	  String delimiter = "x";
	  if (uaPixels == null) {
	    screenRes = xUpDevcapScreenpixels;
	    delimiter = ",";
	  }
	  if (screenRes != null) {
	    String[] resArray = screenRes.split(delimiter);
	    if (resArray.length == 2) {
	      url.append("&u_w=").append(resArray[0]);
	      url.append("&u_h=").append(resArray[1]);
	    }
	  }
	}

	static void googleAppendDcmguid(StringBuilder url, String dcmguid) {
	  if (dcmguid != null) {
	    url.append("&dcmguid=").append(dcmguid);
	  }
	}

	/**
	 * 显示广告
	 * @param req
	 * @param pub_id (pub-1307862221338762)
	 * @param channel (4062021557)
	 * @throws IOException 
	 */
	public static String ad(HttpServletRequest req, String pub_id, String channel) throws IOException {
		
		long googleDt = System.currentTimeMillis();
		String googleHost = (req.isSecure() ? "https://" : "http://") + req.getHeader("Host");

		StringBuilder googleAdUrlStr = new StringBuilder(PAGEAD);
		googleAdUrlStr.append("ad_type=text_image");
		googleAdUrlStr.append("&channel=");
		googleAdUrlStr.append(channel);
		googleAdUrlStr.append("&client=ca-mb-");
		googleAdUrlStr.append(pub_id);
		googleAdUrlStr.append("&dt=").append(googleDt);
		googleAdUrlStr.append("&format=mobile_single");
		googleAppendUrl(googleAdUrlStr, "host", googleHost);
		googleAppendUrl(googleAdUrlStr, "ip", RequestUtils.getRemoteAddr(req));
		googleAdUrlStr.append("&markup=xhtml");
		googleAdUrlStr.append("&oe=utf8");
		googleAdUrlStr.append("&output=xhtml");
		googleAppendUrl(googleAdUrlStr, "ref", req.getHeader("Referer"));
		String googleUrl = req.getRequestURL().toString();
		if (req.getQueryString() != null) {
			googleUrl += "?" + req.getQueryString().toString();
		}
		googleAppendUrl(googleAdUrlStr, "url", googleUrl);
		googleAppendUrl(googleAdUrlStr, "useragent", req
				.getHeader("User-Agent"));
		googleAppendScreenRes(googleAdUrlStr, req.getHeader("UA-pixels"), req
				.getHeader("x-up-devcap-screenpixels"));
		googleAppendDcmguid(googleAdUrlStr, req.getHeader("X-DCMGUID"));

		StringBuilder html = new StringBuilder();
		URL googleAdUrl = new URL(googleAdUrlStr.toString());
		BufferedReader reader = new BufferedReader(new InputStreamReader(
				googleAdUrl.openStream(), "UTF-8"));
		for (String line; (line = reader.readLine()) != null;) {
			html.append(line);
		}
		
		return html.toString();
	}

}
