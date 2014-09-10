package my.cache;

import java.io.*;
import java.util.Hashtable;
import java.util.concurrent.locks.ReentrantLock;

import my.db.DBManager;

import org.apache.commons.lang.StringUtils;
import org.apache.velocity.context.InternalContextAdapter;
import org.apache.velocity.exception.MethodInvocationException;
import org.apache.velocity.exception.ParseErrorException;
import org.apache.velocity.exception.ResourceNotFoundException;
import org.apache.velocity.runtime.directive.Directive;
import org.apache.velocity.runtime.parser.node.Node;
import org.apache.velocity.runtime.parser.node.SimpleNode;

/**
 * Velocity模板上用于控制缓存的指令
 * 该类必须在 velocity.properties 中配置 userdirective=my.cache.CacheDirective
 * @author Winter Lau
 * @date 2009-3-16 下午04:40:19
 */
public class CacheDirective extends Directive {

	private final static Hashtable<String, String> body_templates = new Hashtable<String, String>();
	private final static Hashtable<String, ReentrantLock> g_locks = new Hashtable<String, ReentrantLock>();
	
	@Override
	public String getName() { return "cache"; }
	
	@Override
	public int getType() { return BLOCK; }

	/* (non-Javadoc)
	 * @see Directive#render(InternalContextAdapter, java.io.Writer, Node)
	 */
	@Override
	public boolean render(InternalContextAdapter context, Writer writer, Node node)
			throws IOException, ResourceNotFoundException, ParseErrorException,
			MethodInvocationException 
	{
		//获得缓存信息
        SimpleNode sn_region = (SimpleNode) node.jjtGetChild(0);
        String region = (String)sn_region.value(context);
        SimpleNode sn_key = (SimpleNode) node.jjtGetChild(1);
        Serializable key = (Serializable)sn_key.value(context);
        
        //准备全局锁对象
        String vk = key + "@" + region;
        if(!g_locks.containsKey(vk))
        	g_locks.put(vk, new ReentrantLock());
        
        Node body = node.jjtGetChild(2);
        //检查内容是否有变化        
        String cache_html = executeVmBody(region, key, body, context);
        
        writer.write(cache_html);
        return true;
	}
	
	private String executeVmBody(final String region, final Serializable key,
			final Node body, final InternalContextAdapter context) {
		return (String)ICacheHelper.get(region, key, new ICacheInvoker(){
			public Object callback(Object old_data, Object... args) {
				try{
			        String tpl_key = key+"@"+region;
			        String body_tpl = body.literal();
			        String old_body_tpl = body_templates.get(tpl_key);
			        String cache_html = CacheManager.get(String.class, region, key);
		        	//System.out.println("1 =====================> check cache : " + Thread.currentThread().getName());
			        if(cache_html == null || !StringUtils.equals(body_tpl, old_body_tpl)){
			        	//System.out.println("2 =====================> " + Thread.currentThread().getName());
			        	ReentrantLock lock = g_locks.get(tpl_key);	
			        	//long ct = System.currentTimeMillis();
			        	boolean need_unlock = false;
			        	if(!lock.tryLock()){
				        	//System.out.println("3 =====================> already locked, return old data."+(System.currentTimeMillis()-ct)+":" + Thread.currentThread().getName());
			        		if(old_data != null)
			        			return old_data;
			        	}
			        	else
			        		need_unlock = true;
			        	try{
				        	//System.out.println("4 =====================> execute vm." + Thread.currentThread().getName());
				        	StringWriter sw = new StringWriter();
				        	body.render(context, sw);
				        	cache_html = sw.toString();
				        	CacheManager.set(region, key, cache_html);
				        	body_templates.put(tpl_key, body_tpl);
			        	}finally{
			        		if(need_unlock)
			        			try{
			        			lock.unlock();
			        			}catch(Exception e){}
			        	}
			        }
			        return cache_html;
				}catch(IOException e){
					throw new RuntimeException("IOException when render vm", e);
				}finally{
					DBManager.closeConnection();
				}
			}
		}, body);	
	}
}
