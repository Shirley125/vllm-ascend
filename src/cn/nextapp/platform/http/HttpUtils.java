package cn.nextapp.platform.http;

import java.io.IOException;
import java.io.InputStream;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.ClientProtocolException;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpHead;
import org.apache.http.impl.client.DefaultHttpClient;

/**
 * Http工具包
 * @author Winter Lau
 * @date 2011-12-30 下午12:01:40
 */
public class HttpUtils {

	/**
	 * 检查网址是否可访问
	 * @param url
	 * @return
	 */
	public static boolean isUrlAccessable(String url) {
		DefaultHttpClient client = new DefaultHttpClient();
		try{
			HttpHead head = new HttpHead(url);
			HttpResponse response = client.execute(head);
	        int code = response.getStatusLine().getStatusCode();
	        return (code >= 200 && code < 400);
		}catch(Exception e){
			return false;
		}finally{
			client = null;
		}
	}
	
	/**
	 * 请求URL
	 * @param url
	 * @throws IOException 
	 * @throws ClientProtocolException 
	 */
	public static InputStream httpGet(String url) throws Exception {
		DefaultHttpClient client = new DefaultHttpClient();
		InputStream istream = null;
		try{
			HttpGet get = new HttpGet(url);
	        HttpResponse response = client.execute(get);
	        int code = response.getStatusLine().getStatusCode();
	        if(code != 200)
	        	throw new Exception("http get url failed. status code is " + code);
	        HttpEntity entity = response.getEntity();
	        istream = entity.getContent();
		}finally{
			client = null;
		}
		return istream;
	}
	
}
